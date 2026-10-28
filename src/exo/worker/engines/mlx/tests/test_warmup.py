import pytest

from exo.worker.engines.mlx.generator.warmup import cancellation_check_interval


@pytest.mark.parametrize(
    ("tokens_generated", "elapsed_seconds", "expected_interval"),
    [
        (50, 1500.0, 1),
        (50, 50.0, 1),
        (50, 20.0, 3),
        (50, 5.0, 10),
        (50, 0.5, 100),
        (50, 0.1, 100),
        (1, 0.0, 100),
        (1, 0.0001, 100),
        (0, 0.0, 1),
        (0, 10.0, 1),
    ],
)
def test_cancellation_check_interval(
    tokens_generated: int, elapsed_seconds: float, expected_interval: int
) -> None:
    assert (
        cancellation_check_interval(tokens_generated, elapsed_seconds)
        == expected_interval
    )
