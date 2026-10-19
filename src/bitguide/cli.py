"""命令行入口：`uv run bitguide --help`。

    bitguide agents                                  列出 Agent
    bitguide chat --agent tech_helper -m "..."       单 Agent 对话（不带 -m 进入交互模式）
    bitguide workflow list                           列出 8 个演示工作流
    bitguide workflow run expert_review -i "..."     运行工作流
    bitguide workflow resume <thread> --approve      人工审批后继续（换个进程也行）
    bitguide worker                                  从 Redis 队列取 Gin 网关提交的任务来执行
    bitguide etcd seed                               把 YAML 里的 Agent 定义写进 etcd
加 --offline 可以在没有大模型的机器上跑通全部流程。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx
import openai

from .application import DEFAULT_CONFIG, Platform, load_agents_from_etcd, seed_agents_to_etcd
from .config import PlatformSettings
from .infrastructure import EtcdClient
from .worker import serve
from .workflow_runtime import RunResult

MODEL_HINT = "提示：先 `ollama serve`，再 `ollama pull qwen2.5`；或者加 --offline 用离线模型"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bitguide", description="BitGuide 多 Agent 平台")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG), help="配置文件路径")
    parser.add_argument("--offline", action="store_true", help="使用内置离线模型，不需要 Ollama")
    parser.add_argument("--quiet", action="store_true", help="不打印 ReAct / 工作流执行过程")
    parser.add_argument("--etcd", action="store_true", help="从 etcd 加载 Agent 定义")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("agents", help="列出所有 Agent")

    chat = commands.add_parser("chat", help="和单个 Agent 对话")
    chat.add_argument("--agent", help="Agent id，默认第一个")
    chat.add_argument("-m", "--message", help="只问一句；不填则进入交互模式")
    chat.add_argument("--session", default="cli", help="会话 id，不同会话的记忆互相隔离")

    workflow = commands.add_parser("workflow", help="多 Agent 工作流").add_subparsers(
        dest="action", required=True
    )
    workflow.add_parser("list", help="列出工作流")
    run = workflow.add_parser("run", help="运行工作流")
    run.add_argument("name")
    run.add_argument("-i", "--input", required=True, help="写入 state['input'] 的内容")
    run.add_argument("--thread", help="自定义 thread id")
    resume = workflow.add_parser("resume", help="恢复被中断 / 失败的工作流")
    resume.add_argument("thread")
    decision = resume.add_mutually_exclusive_group()
    decision.add_argument("--approve", action="store_true", help="审批通过")
    decision.add_argument("--reject", action="store_true", help="审批驳回")
    resume.add_argument("--comment", default="", help="审批意见")
    workflow.add_parser("show", help="查看某次运行的 Checkpoint").add_argument("thread")
    workflow.add_parser("history", help="最近的运行记录")

    worker = commands.add_parser("worker", help="执行 Gin 网关提交的工作流任务")
    worker.add_argument("--redis", default="redis://localhost:6379/0", help="任务队列所在的 Redis")
    worker.add_argument("--concurrency", type=int, default=4, help="最多同时执行几个任务")

    etcd = commands.add_parser("etcd", help="etcd 配置中心").add_subparsers(
        dest="action", required=True
    )
    etcd.add_parser("seed", help="把 YAML 里的 Agent 定义写进 etcd")
    etcd.add_parser("list", help="列出 etcd 里的 Agent 定义")
    return parser


def print_result(result: RunResult) -> None:
    print(
        f"\n工作流 {result.workflow} · thread={result.thread_id}"
        f" · {result.status} · {result.steps} 步"
    )
    if result.status == "completed":
        print(f"\n{result.output}")
    elif result.status == "interrupted" and result.interrupt:
        payload = result.interrupt.get("payload") or {}
        print(f"\n⏸  {payload.get('question', '等待人工输入')}\n{payload.get('preview', '')}")
        print(f"\n继续：bitguide workflow resume {result.thread_id} --approve")
        print(f'驳回：bitguide workflow resume {result.thread_id} --reject --comment "修改意见"')
    elif result.status == "failed":
        print(f"\n❌ {result.error}")
        if any(word in (result.error or "") for word in ("Connection", "NotFound")):
            print(MODEL_HINT)
        print(f"修好后可以重试：bitguide workflow resume {result.thread_id}")


async def _agents(platform: Platform) -> None:
    for info in await platform.agents.infos():
        async with platform.agents.acquire(info.id) as agent:
            tools = ", ".join(agent.tool_names()) or "-"
        print(f"{info.id:<18} v{info.version}  {info.name}  工具: {tools}")
        if info.description:
            print(f"{'':<18}      {info.description}")


async def _chat(platform: Platform, args: argparse.Namespace) -> None:
    ids = await platform.agents.ids()
    # 默认用配置文件里写在最前面的 Agent
    agent_id = args.agent or next((a for a in platform.settings.agents if a in ids), ids[0])
    if args.message:
        print((await platform.chat(agent_id, args.message, args.session)).content)
        return
    print(f"和 {agent_id} 对话。/clear 清空记忆，/exit 退出。")
    while True:
        try:
            line = (await asyncio.to_thread(input, "\n你> ")).strip()
        except EOFError:
            break
        if line in ("/exit", "/quit"):
            break
        if line == "/clear":
            async with platform.agents.acquire(agent_id) as agent:
                await agent.clear_session(args.session)
            print("（记忆已清空）")
            continue
        if line:
            response = await platform.chat(agent_id, line, args.session)
            print(f"\n{agent_id}> {response.content}")


async def _workflow(platform: Platform, args: argparse.Namespace) -> None:
    service = platform.workflows
    if args.action == "list":
        for graph in service.registry.all():
            print(f"{graph.name:<18} {graph.description}")
    elif args.action == "run":
        print_result(await service.run(args.name, {"input": args.input}, args.thread))
    elif args.action == "resume":
        value = None
        if args.approve or args.reject:
            value = {"approved": args.approve, "comment": args.comment}
        checkpoint = await service.get(args.thread)
        if checkpoint and checkpoint.status == "interrupted" and value is None:
            raise ValueError("这个工作流在等人工审批，请加 --approve 或 --reject")
        print_result(await service.resume(args.thread, value))
    elif args.action == "show":
        checkpoint = await service.get(args.thread)
        if checkpoint is None:
            raise KeyError(f"checkpoint not found: {args.thread}")
        print(json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2))
    elif args.action == "history":
        for cp in await service.history():
            print(
                f"{cp.thread_id}  {cp.workflow:<18} {cp.status:<12} "
                f"第 {cp.step} 步  {cp.updated_at}"
            )


async def _etcd(args: argparse.Namespace) -> None:
    settings = PlatformSettings.load(args.config)
    etcd = EtcdClient(settings.etcd.endpoint, settings.etcd.timeout_seconds)
    try:
        if args.action == "seed":
            await seed_agents_to_etcd(etcd, settings.etcd.prefix, list(settings.agents.values()))
            print(f"已写入 {len(settings.agents)} 个 Agent 到 {settings.etcd.prefix}")
        else:
            for agent_id, spec in (await load_agents_from_etcd(etcd, settings.etcd.prefix)).items():
                print(f"{agent_id:<18} model={spec.model_ref} tools={spec.tools}")
    finally:
        await etcd.close()


async def _dispatch(args: argparse.Namespace) -> None:
    if args.command == "etcd":
        await _etcd(args)
        return
    # 只查看记录时不需要装配 Agent（省掉起 MCP 子进程、建索引的时间）
    needs_agents = not (args.command == "workflow" and args.action in ("list", "show", "history"))
    platform = await Platform.create(
        args.config,
        offline=args.offline,
        trace=False if args.quiet else None,
        use_etcd=True if args.etcd else None,
        only_agents=None if needs_agents else [],
    )
    async with platform:
        if args.command == "agents":
            await _agents(platform)
        elif args.command == "chat":
            await _chat(platform, args)
        elif args.command == "worker":
            await serve(platform.workflows, args.redis, args.concurrency)
        else:
            await _workflow(platform, args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(_dispatch(args))
    except openai.APIConnectionError:
        print(f"连不上大模型服务。{MODEL_HINT}", file=sys.stderr)
        return 2
    except openai.APIStatusError as error:
        print(
            f"模型服务返回 HTTP {error.status_code}：{error.message}\n{MODEL_HINT}", file=sys.stderr
        )
        return 2
    except (KeyError, ValueError, OSError, httpx.HTTPError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
