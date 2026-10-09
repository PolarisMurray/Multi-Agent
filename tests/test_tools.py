import dataclasses
import re

import pytest

from bitguide.tools import (
    BUILTIN_TOOLS,
    CalculatorTool,
    DataAnalyzerTool,
    DateTimeTool,
    ToolManager,
    ToolResult,
    WebSearchTool,
    build_tool_manager,
)

# ---------- ToolResult ----------


def test_tool_result_ok_and_fail():
    ok = ToolResult.ok("42")
    assert ok.success is True and ok.content == "42" and ok.error is None
    bad = ToolResult.fail("boom")
    assert bad.success is False and bad.content == "" and bad.error == "boom"


def test_tool_result_for_model():
    assert ToolResult.ok("42").for_model() == "42"
    # 失败也要变成一段模型能读懂的文字，而不是抛异常
    assert "boom" in ToolResult.fail("boom").for_model()


def test_tool_result_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ToolResult.ok("x").content = "y"  # type: ignore[misc]


# ---------- ToolManager：注册 ----------


class EchoTool:
    name = "echo"
    description = "echo back"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    async def invoke(self, arguments):
        return ToolResult.ok(arguments["text"])


class RawReturnTool:
    """返回的不是 ToolResult，Manager 要帮忙包一层。"""

    name = "raw"
    description = "returns a plain int"
    parameters = {"type": "object", "properties": {}}

    async def invoke(self, arguments):
        return 7


class ExplodingTool:
    name = "explode"
    description = "always raises"
    parameters = {"type": "object", "properties": {}}

    async def invoke(self, arguments):
        raise RuntimeError("disk on fire")


def test_register_and_names_sorted():
    manager = ToolManager()
    manager.register(RawReturnTool())
    manager.register(EchoTool())
    assert manager.names() == ["echo", "raw"]
    assert manager.get("echo") is not None
    assert manager.get("missing") is None
    assert [t.name for t in manager.values()] == ["echo", "raw"]


def test_register_duplicate_raises_unless_replace():
    manager = ToolManager()
    manager.register(EchoTool())
    with pytest.raises(ValueError):
        manager.register(EchoTool())
    replacement = EchoTool()
    manager.register(replacement, replace=True)
    assert manager.get("echo") is replacement


# ---------- ToolManager：异常隔离（本课重点之一） ----------


async def test_invoke_success():
    manager = ToolManager()
    manager.register(EchoTool())
    result = await manager.invoke("echo", {"text": "hi"})
    assert result == ToolResult.ok("hi")


async def test_invoke_unknown_tool_returns_fail_not_raise():
    result = await ToolManager().invoke("nope", {})
    assert result.success is False
    assert "nope" in (result.error or "")


async def test_invoke_exception_is_isolated():
    manager = ToolManager()
    manager.register(ExplodingTool())
    result = await manager.invoke("explode", {})  # 不能抛出来
    assert result.success is False
    assert "disk on fire" in (result.error or "")


async def test_invoke_wraps_non_tool_result():
    manager = ToolManager()
    manager.register(RawReturnTool())
    assert await manager.invoke("raw", {}) == ToolResult.ok("7")


# ---------- CalculatorTool：AST 安全计算器（本课重点之二） ----------


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("1 + 2", "3"),
        ("2 + 3 * 4", "14"),  # 优先级
        ("(2 + 3) * 4", "20"),  # 括号
        ("-5 + 2", "-3"),  # 一元负号
        ("7 % 3", "1"),
        ("7 / 2", "3.5"),
    ],
)
async def test_calculator_valid(expr, expected):
    result = await CalculatorTool().invoke({"expression": expr})
    assert result.success, result.error
    assert result.content.endswith(f"= {expected}")


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo pwned')",  # 函数调用 —— eval 的经典漏洞
        "().__class__.__bases__[0].__subclasses__()",  # 属性访问逃逸沙箱
        "x + 1",  # 变量名
        "'a' * 3",  # 字符串常量
        "True + 1",  # bool 是 int 的子类，也要拒绝
        "9 ** 9 ** 9",  # 幂运算：不支持，防止 CPU/内存打爆
        "[1, 2]",
        "lambda: 1",
    ],
)
async def test_calculator_rejects_unsafe(expr):
    result = await CalculatorTool().invoke({"expression": expr})
    assert result.success is False


@pytest.mark.parametrize("expr", ["", "   ", "1 / 0", "1 +"])
async def test_calculator_bad_input_fails_gracefully(expr):
    result = await CalculatorTool().invoke({"expression": expr})
    assert result.success is False


async def test_calculator_missing_argument():
    assert (await CalculatorTool().invoke({})).success is False


# ---------- 其他内置工具 ----------


async def test_datetime_formats():
    tool = DateTimeTool()
    date = await tool.invoke({"format": "date"})
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date.content)
    time = await tool.invoke({"format": "time"})
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", time.content)
    assert (await tool.invoke({})).success  # 默认 datetime
    assert (await tool.invoke({"format": "bogus"})).success is False


async def test_web_search_is_simulated():
    tool = WebSearchTool()
    result = await tool.invoke({"query": "python", "top_k": 2})
    assert result.success
    assert result.content.count("python") >= 2
    assert (await tool.invoke({"query": ""})).success is False


async def test_data_analyzer():
    tool = DataAnalyzerTool()
    result = await tool.invoke({"values": [1, 2, 3, 4]})
    assert result.success
    for part in ["count=4", "sum=10", "mean=2.5", "median=2.5", "min=1", "max=4"]:
        assert part in result.content
    assert (await tool.invoke({"values": []})).success is False
    assert (await tool.invoke({"values": [1, "x"]})).success is False
    assert (await tool.invoke({})).success is False


# ---------- 工具 schema 与装配 ----------


@pytest.mark.parametrize("cls", list(BUILTIN_TOOLS.values()))
def test_builtin_tools_expose_json_schema(cls):
    tool = cls()
    assert isinstance(tool.name, str) and tool.name
    assert isinstance(tool.description, str) and tool.description
    assert tool.parameters["type"] == "object"


def test_build_tool_manager():
    manager = build_tool_manager(["calculator", "datetime"])
    assert manager.names() == ["calculator", "datetime"]
    with pytest.raises(ValueError):
        build_tool_manager(["calculator", "teleport"])
