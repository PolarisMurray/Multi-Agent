from bitguide.contracts import LLMClient, Tool
from bitguide.models import Message, ModelTurn
from bitguide.tools import CalculatorTool


class EchoLLM:  # 注意：没有继承 LLMClient
    async def complete(self, messages, tools):
        return ModelTurn(content=messages[-1].content)


async def test_fake_llm_satisfies_protocol():
    llm: LLMClient = EchoLLM()  # pyright 会在这里检查结构是否匹配
    turn = await llm.complete([Message("user", "hi")], [])
    assert turn.content == "hi"


def test_builtin_tool_satisfies_protocol():
    tool: Tool = CalculatorTool()
    assert tool.name == "calculator"
