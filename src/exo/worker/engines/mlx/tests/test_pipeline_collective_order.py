from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from inspect import Signature, signature
from pathlib import Path
from typing import Self, cast

import pytest


@dataclass
class FakeArray:
    name: str
    events: list[str]
    shape: tuple[int, ...] = (1, 1, 2048)
    evaluated: bool = False

    def __getitem__(self, key: slice) -> Self:
        assert self.evaluated, "Collective must be evaluated before slicing"
        assert key == slice(-1, None)
        self.events.append(f"slice:{self.name}")
        return type(self)("sliced", self.events)


@dataclass
class FakeDistributed:
    events: list[str]

    def send(self, output: FakeArray, destination: int, *, group: object) -> FakeArray:
        assert output.evaluated
        assert destination == 1
        self.events.append("send")
        return FakeArray("sent", self.events)

    def all_gather(self, output: FakeArray, *, group: object) -> FakeArray:
        assert output.evaluated
        self.events.append("all_gather")
        return FakeArray("gathered", self.events, shape=(2, 1, 2048))


@dataclass
class FakeCore:
    distributed: FakeDistributed

    def eval(self, array: FakeArray) -> None:
        array.events.append(f"eval:{array.name}")
        array.evaluated = True


@dataclass
class FakePipelineLastLayer:
    r: int
    is_prefill: bool
    group: object
    s: int = 2
    queue_sends: bool = False

    @staticmethod
    def original_layer(x: FakeArray) -> FakeArray:
        return x

    @property
    def original_layer_signature(self) -> Signature:
        return signature(self.original_layer)


def pipeline_last_layer_call(
    core: FakeCore,
) -> Callable[[FakePipelineLastLayer, FakeArray], FakeArray]:
    """Execute the actual method with fakes without importing MLX or model modules."""
    source_path = Path(__file__).parents[1] / "auto_parallel.py"
    module = ast.parse(source_path.read_text())
    layer = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "PipelineLastLayer"
    )
    method = next(
        node
        for node in layer.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    namespace: dict[str, object] = {"mx": core}
    code = compile(
        ast.Module(body=[method], type_ignores=[]),
        str(source_path),
        "exec",
    )
    exec(code, namespace)
    return cast(
        Callable[[FakePipelineLastLayer, FakeArray], FakeArray], namespace["__call__"]
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("is_prefill", [False, True])
def test_pipeline_collective_evaluated_before_slice(
    rank: int, is_prefill: bool
) -> None:
    events: list[str] = []
    core = FakeCore(FakeDistributed(events))
    layer = FakePipelineLastLayer(rank, is_prefill, object())
    output = pipeline_last_layer_call(core)(layer, FakeArray("layer", events))

    expected = ["eval:layer"]
    if rank == 0:
        expected.extend(["send", "eval:sent"])
    if not is_prefill:
        expected.extend(
            ["all_gather", "eval:gathered", "slice:gathered", "eval:sliced"]
        )
    assert events == expected
    expected_output = "sliced" if not is_prefill else "sent" if rank == 0 else "layer"
    assert output.name == expected_output
