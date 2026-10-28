import math


def cancellation_check_interval(tokens_generated: int, elapsed_seconds: float) -> int:
    """Target one cancellation check per second, bounded to 1-100 tokens."""
    tokens_per_second = tokens_generated / max(elapsed_seconds, 0.001)
    return max(1, min(math.ceil(tokens_per_second), 100))
