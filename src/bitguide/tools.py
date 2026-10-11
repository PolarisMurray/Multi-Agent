from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True, frozen=True)
class ToolResult:
    success: bool
    content: str = ""
    error: str | None = None

    @classmethod
    def ok(cls, content: str) -> ToolResult:
        return cls(success=True, content=content)

    @classmethod
    def fail(cls, error: str) -> ToolResult:
        return cls(success=False, error=error)

    def for_model(self) -> str:
        if self.success:
            return self.content
        return f"Tool execution failed: {self.error}"


class ToolManager:
    """Registry and exception boundary: the agent never calls a tool object directly."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool: Any, *, replace: bool = False) -> None:
        name = tool.name
        if name in self._tools and not replace:
            raise ValueError(f"Tool '{name}' already exists")
        self._tools[name] = tool

    def get(self, name: str) -> Any | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def values(self) -> list[Any]:
        return [self._tools[name] for name in self.names()]

    async def invoke(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult.fail(f"Tool '{name}' does not exist")
        try:
            result = await tool.invoke(arguments)
        except Exception as error:
            
            return ToolResult.fail(str(error))
        
        return result if isinstance(result, ToolResult) else ToolResult.ok(str(result))


class CalculatorTool:
    name = "calculator"
    description = "Evaluate arithmetic expressions safely (+ - * / // % ** and parentheses)."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "A mathematical expression to evaluate.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    _operators: dict[type[ast.AST], Callable[..., Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _evaluate(self, node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return self._evaluate(node.body)

        if isinstance(node, ast.Constant):

            if type(node.value) not in (int, float):
                raise ValueError("Only numbers are allowed")
            return node.value

        if isinstance(node, ast.BinOp):
            operation = self._operators.get(type(node.op))
            if operation is None:
                raise ValueError("Unsupported operator")
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1_000_000):
                raise ValueError("Exponentiation is too large")
            result = operation(left, right)
            
            if type(result) not in (int, float):
                raise ValueError("Result must be a real number")
            return result

        if isinstance(node, ast.UnaryOp):
            operation = self._operators.get(type(node.op))
            if operation is None:
                raise ValueError("Unsupported unary operator")
            return operation(self._evaluate(node.operand))

        raise ValueError("Unsupported expression")

    async def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        expression = arguments.get("expression", "")
        if not isinstance(expression, str) or not expression.strip():
            return ToolResult.fail("Expression cannot be empty")
        if len(expression) > 200:
            return ToolResult.fail("Expression is too long")
        try:
            result = self._evaluate(ast.parse(expression, mode="eval"))
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as error:
            return ToolResult.fail(str(error))
    
        return ToolResult.ok(f"{expression} = {result}")


class DateTimeTool:
    name = "datetime"
    description = "Get the current date, time, weekday, or ISO timestamp."
    parameters = {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["datetime", "date", "time", "weekday", "iso"],
                "default": "datetime",
            }
        },
        "additionalProperties": False,
    }
    _formats = {
        "datetime": "%Y-%m-%d %H:%M:%S %Z",
        "date": "%Y-%m-%d",
        "time": "%H:%M:%S",
        "weekday": "%A",
    }

    async def invoke(self, arguments: Mapping[str, Any]) -> ToolResult:
        now = datetime.now().astimezone()  # 带本地时区，避免"几点"产生歧义
        fmt = str(arguments.get("format", "datetime"))
        if fmt == "iso":
            return ToolResult.ok(now.isoformat())
        if fmt not in self._formats:
            return ToolResult.fail(f"Unsupported format: {fmt}")
        return ToolResult.ok(now.strftime(self._formats[fmt]))
