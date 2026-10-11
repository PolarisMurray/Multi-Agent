import pytest

from bitguide.tools import CalculatorTool, DateTimeTool, ToolManager, ToolResult


class BoomTool:
    name = "boom"
    description = "always raises"
    parameters = {"type": "object", "properties": {}}

    async def invoke(self, arguments):
        raise RuntimeError("boom")


class PlainStringTool:
    name = "plain"
    description = "returns a bare string instead of ToolResult"
    parameters = {"type": "object", "properties": {}}

    async def invoke(self, arguments):
        return "hello"


# ---------- ToolResult ----------


def test_tool_result_for_model():
    assert ToolResult.ok("42").for_model() == "42"
    assert "Divide by zero" in ToolResult.fail("Divide by zero").for_model()


def test_tool_result_is_immutable():
    result = ToolResult.ok("42")
    with pytest.raises(AttributeError):
        result.content = "changed"  # type: ignore[misc]


# ---------- ToolManager ----------


def test_manager_rejects_duplicate():
    m = ToolManager()
    m.register(CalculatorTool())
    with pytest.raises(ValueError):
        m.register(CalculatorTool())
    m.register(CalculatorTool(), replace=True)


def test_names_and_values_are_sorted():
    m = ToolManager()
    m.register(DateTimeTool())
    m.register(CalculatorTool())
    assert m.names() == ["calculator", "datetime"]
    assert [tool.name for tool in m.values()] == ["calculator", "datetime"]


async def test_manager_isolates_exceptions():
    m = ToolManager()
    m.register(BoomTool())
    result = await m.invoke("boom", {})
    assert isinstance(result, ToolResult)
    assert not result.success
    assert "boom" in (result.error or "")


async def test_unknown_tool_is_observation_not_crash():
    result = await ToolManager().invoke("nope", {})
    assert isinstance(result, ToolResult)
    assert not result.success


async def test_manager_wraps_plain_return_value():
    m = ToolManager()
    m.register(PlainStringTool())
    result = await m.invoke("plain", {})
    assert result == ToolResult.ok("hello")


# ---------- CalculatorTool ----------


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("1 + 2 * 3", "7"),
        ("(1 + 2) * 3", "9"),
        ("-4 % 3", "2"),
        ("7 / 2", "3.5"),
        ("7 // 2", "3"),
        ("2 ** 10", "1024"),
    ],
)
async def test_calculator_math(expr, expected):
    r = await CalculatorTool().invoke({"expression": expr})
    assert r.success, r.error
    assert r.content.endswith(expected)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",  # 函数调用：代码注入
        "2 ** 100000000",  # 超大指数：DoS
        "(-8) ** 0.5",  # 复数结果
        "True + 1",  # bool 伪装成 int
        "'a' * 3",  # 字符串
        "x + 1",  # 变量名
        "1 / 0",
        "1 +",  # 语法错误
        "",
        "   ",
        "1+" * 150 + "1",  # 超长
    ],
)
async def test_calculator_rejects_dangerous_or_invalid(expr):
    r = await CalculatorTool().invoke({"expression": expr})
    assert not r.success


async def test_calculator_rejects_non_string():
    r = await CalculatorTool().invoke({"expression": 123})
    assert not r.success


# ---------- DateTimeTool ----------


async def test_datetime_formats():
    t = DateTimeTool()
    r = await t.invoke({"format": "date"})
    assert r.success and len(r.content) == 10
    assert (await t.invoke({})).success  # 默认 datetime
    assert "T" in (await t.invoke({"format": "iso"})).content
    assert not (await t.invoke({"format": "century"})).success
