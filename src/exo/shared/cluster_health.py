from collections.abc import Mapping
from datetime import datetime
from typing import Literal

from exo.shared.election import MASTER_SILENCE_TIMEOUT
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.state import State
from exo.utils.pydantic_ext import FrozenModel


class ClusterHealth(FrozenModel):
    scope: Literal["local_observation"] = "local_observation"
    status: Literal["healthy", "degraded", "recovering"]
    session_id: SessionId | None = None
    issues: list[str] = []
    unavailable_node_ids: list[NodeId] = []
    master_silence_seconds: float | None = None
    state_silence_seconds: float | None = None
    retry_in_seconds: float = 0.0
    missing_event_idx: int | None = None


def cluster_health(
    state: State,
    session_id: SessionId,
    *,
    now: datetime,
    monotonic_now: float,
    campaign_running: bool,
    awaiting_snapshot: bool,
    master_silence_seconds: float | None,
    master_confirmed: bool,
    last_delivery_at: float | None,
    missing_event_idx: int | None,
    conflicting_masters: Mapping[NodeId, float],
    node_last_seen: Mapping[NodeId, float],
    retry_in_seconds: float,
) -> ClusterHealth:
    issues: list[str] = []
    unavailable = set(state.node_timeouts)
    unavailable.update(
        node_id
        for node_id, seen in state.last_seen.items()
        if (now - seen).total_seconds() > MASTER_SILENCE_TIMEOUT
    )
    # A heartbeat from an old session is not recovery. Require a later report
    # delivered through the current session's ordered event stream.
    unavailable.update(
        node_id
        for node_id, heard_at in conflicting_masters.items()
        if node_last_seen.get(node_id, -1.0) <= heard_at
    )
    if unavailable:
        issues.append("Peers are missing or have not rejoined the current session")
    if (
        master_silence_seconds is not None
        and master_silence_seconds > MASTER_SILENCE_TIMEOUT
    ):
        issues.append(
            "Master heartbeat timed out; automatic election retries are active"
        )
    degraded = bool(issues)
    if not master_confirmed:
        issues.append("Waiting for a heartbeat from the current master")
    if campaign_running:
        issues.append("Master election in progress")
    if awaiting_snapshot:
        issues.append("Waiting for the current master's state snapshot")
    if missing_event_idx is not None:
        issues.append("Cluster state has an event gap; replay retries are active")
    state_silence = (
        max(0.0, monotonic_now - last_delivery_at)
        if last_delivery_at is not None
        else None
    )
    if state_silence is None or state_silence > MASTER_SILENCE_TIMEOUT:
        issues.append("Waiting for fresh current-session state")
    return ClusterHealth(
        status="degraded" if degraded else "recovering" if issues else "healthy",
        session_id=session_id,
        issues=issues,
        unavailable_node_ids=sorted(unavailable),
        master_silence_seconds=master_silence_seconds,
        state_silence_seconds=state_silence,
        retry_in_seconds=retry_in_seconds,
        missing_event_idx=missing_event_idx,
    )
