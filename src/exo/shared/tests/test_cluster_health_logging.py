import pytest

from exo.main import Node
from exo.shared.cluster_health import ClusterHealth


class MonitorStoppedError(Exception):
    pass


async def test_health_warnings_are_rate_limited_and_recovery_logged_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    warnings: list[float] = []
    recoveries: list[float] = []

    def health(_node: Node) -> ClusterHealth:
        return ClusterHealth(
            status="degraded" if now < 65 else "healthy",
            issues=["Peer missing"] if now < 65 else [],
        )

    async def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds
        if now >= 80:
            raise MonitorStoppedError

    def warning(_message: str) -> None:
        warnings.append(now)

    def recovered(_message: str) -> None:
        recoveries.append(now)

    monkeypatch.setattr(Node, "cluster_health", health)
    monkeypatch.setattr("exo.main.anyio.current_time", lambda: now)
    monkeypatch.setattr("exo.main.anyio.sleep", sleep)
    monkeypatch.setattr("exo.main.logger.warning", warning)
    monkeypatch.setattr("exo.main.logger.info", recovered)
    node = object.__new__(Node)
    with pytest.raises(MonitorStoppedError):
        await node._monitor_cluster_health()  # pyright: ignore[reportPrivateUsage]
    assert warnings == [0.0, 30.0, 60.0]
    assert recoveries == [65.0]
