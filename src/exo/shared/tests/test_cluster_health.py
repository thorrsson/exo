from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

import pytest

from exo.shared.apply import apply_node_gathered_info, apply_node_timed_out
from exo.shared.cluster_health import ClusterHealth, cluster_health
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import NodeGatheredInfo, NodeTimedOut
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State

MASTER = NodeId("master")
PEER = NodeId("peer")
SESSION = SessionId(master_node_id=MASTER, election_clock=4)
NOW = datetime.now(timezone.utc)


def health(
    state: State | None = None,
    *,
    campaign_running: bool = False,
    awaiting_snapshot: bool = False,
    master_confirmed: bool = True,
    master_silence_seconds: float | None = 1.0,
    last_delivery_at: float | None = 99.0,
    missing_event_idx: int | None = None,
    conflicting_masters: Mapping[NodeId, float] | None = None,
    node_last_seen: Mapping[NodeId, float] | None = None,
) -> ClusterHealth:
    return cluster_health(
        state or State(),
        SESSION,
        now=NOW,
        monotonic_now=100.0,
        campaign_running=campaign_running,
        awaiting_snapshot=awaiting_snapshot,
        master_confirmed=master_confirmed,
        master_silence_seconds=master_silence_seconds,
        last_delivery_at=last_delivery_at,
        missing_event_idx=missing_event_idx,
        conflicting_masters=conflicting_masters or {},
        node_last_seen=node_last_seen or {},
        retry_in_seconds=15.0,
    )


def test_temporary_silence_does_not_report_dropout() -> None:
    state = State(last_seen={MASTER: NOW, PEER: NOW - timedelta(seconds=29)})
    assert health(state, master_silence_seconds=29.0).status == "healthy"


def test_silent_peer_is_visible_before_removal_and_after_snapshot() -> None:
    state = State(last_seen={MASTER: NOW, PEER: NOW - timedelta(seconds=61)})
    assert health(state).unavailable_node_ids == [PEER]
    removed = apply_node_timed_out(NodeTimedOut(node_id=PEER), state)
    restored = State.model_validate_json(removed.model_dump_json())
    assert PEER not in restored.topology.list_nodes()
    assert PEER not in restored.last_seen
    assert health(restored).status == "degraded"
    assert health(restored).unavailable_node_ids == [PEER]


def test_old_session_return_requires_current_session_report_to_recover() -> None:
    state = State(node_timeouts={PEER: NOW - timedelta(seconds=61)})
    assert health(state, conflicting_masters={PEER: 90.0}).status == "degraded"
    recovered = apply_node_gathered_info(
        NodeGatheredInfo(
            node_id=PEER,
            when=NOW.isoformat(),
            info=MemoryUsage.from_bytes(
                ram_total=1024, ram_available=512, swap_total=0, swap_available=0
            ),
        ),
        state,
    )
    assert not recovered.node_timeouts
    # Even a fresh snapshot does not prove that the conflicting peer has rejoined.
    assert health(recovered, conflicting_masters={PEER: 90.0}).status == "degraded"
    assert (
        health(
            recovered,
            conflicting_masters={PEER: 90.0},
            node_last_seen={PEER: 99.0},
        ).status
        == "healthy"
    )


def test_stale_node_report_does_not_clear_a_retained_timeout() -> None:
    state = State(node_timeouts={PEER: NOW})
    unchanged = apply_node_gathered_info(
        NodeGatheredInfo(
            node_id=PEER,
            when=(NOW - timedelta(seconds=1)).isoformat(),
            info=MemoryUsage.from_bytes(
                ram_total=1024, ram_available=512, swap_total=0, swap_available=0
            ),
        ),
        state,
    )
    assert unchanged.node_timeouts == state.node_timeouts
    assert health(unchanged).status == "degraded"


@pytest.mark.parametrize(
    ("campaign", "snapshot", "confirmed", "last_delivery", "missing"),
    [
        (True, False, True, 99.0, None),
        (False, True, True, 99.0, None),
        (False, False, False, 99.0, None),
        (False, False, True, None, None),
        (False, False, True, 60.0, None),
        (False, False, True, 99.0, 42),
    ],
)
def test_health_waits_for_current_session_recovery(
    campaign: bool,
    snapshot: bool,
    confirmed: bool,
    last_delivery: float | None,
    missing: int | None,
) -> None:
    result = health(
        campaign_running=campaign,
        awaiting_snapshot=snapshot,
        master_confirmed=confirmed,
        last_delivery_at=last_delivery,
        missing_event_idx=missing,
    )
    assert result.status == "recovering"
    assert result.issues


def test_silent_master_stays_degraded_during_retry_backoff() -> None:
    result = health(master_silence_seconds=31.0)
    assert result.status == "degraded"
    assert result.retry_in_seconds == 15.0
    assert result.master_silence_seconds == 31.0
