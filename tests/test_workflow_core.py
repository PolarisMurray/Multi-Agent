import pytest

from bitguide.workflow_core import (
    END,
    START,
    Append,
    ConditionalEdge,
    Ephemeral,
    Graph,
    InvalidUpdateError,
    LastValue,
    apply_writes,
)
from bitguide.workflow_nodes import FunctionNode


def node(name: str) -> FunctionNode:
    return FunctionNode(name, lambda state: None)


def linear() -> Graph:
    return Graph("g").add_node(node("a")).add_edge(START, "a").add_edge("a", END)


def test_valid_graph_passes():
    assert linear().validate().name == "g"


@pytest.mark.parametrize(
    ("graph", "message"),
    [
        (Graph("g"), "no nodes"),
        (Graph("g").add_node(node("a")).add_edge("a", END), "missing edge from START"),
        (linear().add_edge("a", "ghost"), "unknown edge target"),
        (linear().add_edge("ghost", "a"), "unknown edge source"),
        (linear().add_node(node("b")).add_edge("b", END), "unreachable"),
        (Graph("g").add_node(node("a")).add_edge(START, "a"), "no outgoing edge"),
        (
            Graph("g")
            .add_node(node("a"))
            .add_node(node("b"))
            .add_edge(START, "a")
            .add_edge("a", "b")
            .add_edge("b", "a"),
            "END is unreachable",
        ),
    ],
)
def test_structural_errors_are_caught_before_running(graph, message):
    with pytest.raises(ValueError, match=message):
        graph.validate()


def test_reserved_and_duplicate_node_names():
    with pytest.raises(ValueError, match="reserved"):
        Graph("g").add_node(node(END))
    with pytest.raises(ValueError, match="duplicate"):
        Graph("g").add_node(node("a")).add_node(node("a"))


def test_conditional_edge_resolves_through_mapping():
    edge = ConditionalEdge("a", lambda s: s["route"], {"x": "node_x", "y": "node_y"})
    assert edge.resolve({"route": "y"}) == ["node_y"]
    assert edge.possible_targets() == {"node_x", "node_y"}
    with pytest.raises(KeyError, match="unknown branch"):
        edge.resolve({"route": "z"})


def test_next_nodes_dedupes_fan_in_and_drops_end():
    graph = (
        Graph("g")
        .add_node(node("a"))
        .add_node(node("b"))
        .add_node(node("join"))
        .add_edge(START, "a")
        .add_edge(START, "b")
        .add_edge("a", "join")
        .add_edge("b", "join")
        .add_edge("b", END)
        .add_edge("join", END)
    )
    assert graph.next_nodes([START], {}) == ["a", "b"]
    assert graph.next_nodes(["a", "b"], {}) == ["join"]


def test_last_value_overwrites_but_rejects_parallel_conflicts():
    assert apply_writes({}, {"x": 1}, [{"x": 2}]) == {"x": 2}
    with pytest.raises(InvalidUpdateError, match="Append"):
        apply_writes({}, {}, [{"x": 1}, {"x": 2}])
    assert LastValue().reduce(None, [3]) == 3


def test_append_collects_writes_in_order():
    channels = {"notes": Append()}
    state = apply_writes(channels, {"notes": ["old"]}, [{"notes": "a"}, {"notes": ["b", "c"]}])
    assert state["notes"] == ["old", "a", "b", "c"]


def test_ephemeral_value_lives_for_exactly_one_step():
    channels = {"hint": Ephemeral()}
    step1 = apply_writes(channels, {}, [{"hint": "only once", "x": 1}])
    assert step1 == {"hint": "only once", "x": 1}
    step2 = apply_writes(channels, step1, [{"y": 2}])
    assert step2 == {"x": 1, "y": 2}


def test_apply_writes_never_mutates_previous_state():
    old = {"notes": ["a"]}
    new = apply_writes({"notes": Append()}, old, [{"notes": "b"}])
    assert old == {"notes": ["a"]}
    assert new == {"notes": ["a", "b"]}
