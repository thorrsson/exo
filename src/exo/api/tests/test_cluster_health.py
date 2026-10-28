import anyio
import httpx
from fastapi import FastAPI

from exo.api.main import API
from exo.shared.cluster_health import ClusterHealth


async def test_health_endpoint_runs_provider_on_event_loop_and_uses_camel_case() -> (
    None
):
    api = object.__new__(API)

    def provider() -> ClusterHealth:
        # Node health uses the event loop's monotonic clock, not a worker thread.
        assert anyio.current_time() > 0
        return ClusterHealth(
            status="recovering",
            issues=["Waiting for current-session state"],
            missing_event_idx=42,
        )

    api.cluster_health_provider = provider
    app = FastAPI()
    app.get("/cluster/health")(api.get_cluster_health)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/cluster/health")
    assert response.status_code == 200
    assert response.json()["status"] == "recovering"
    assert response.json()["missingEventIdx"] == 42


async def test_missing_health_provider_is_not_reported_healthy() -> None:
    api = object.__new__(API)
    api.cluster_health_provider = None
    assert (await api.get_cluster_health()).status == "recovering"
