import pytest

from bitguide.agent import Agent
from bitguide.agent import ConversationMemory
from bitguide.agent import AgentRequest, ModelTurn, ToolCall
from bitguide.agent import CalculatorTool

class ScriptedLLM:
    """return each ModelTurn by steps, and record, save the tools and messages that each call used"""

    def __init__ (self, *turn: ModelTurn):
        self._turns = list(turns)
        self.call : list[list] = []
        self.tools_seen : list[list] = []

    async def complete(self, messages, tools):
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        return self._turns.pop(0)



class FixedRetriever:
    async def retrieve(self, query, top_k = 5):
        return ["5 year old kid should not plauy cod"]


def calc(call_id :str, expr : str) -> ToolCall:
    return ToolCall(call_id, "calculator", {"expression": expr})

def req(text: str, session: str = "s1") -> AgentRequest:
    return AgentRequest(message = text, sesion_id = session)

