"""A node stops its old master before it follows a new one."""

from typing import cast

import anyio
import pytest

from exo.api.main import API
from exo.download.tests.test_download_status_not_lost import (
    MODEL_ID,
    NODE_ID,
    SHARD,
    FakeShardDownloader,
    _setup_coordinator,  # pyright: ignore[reportPrivateUsage]
)
from exo.main import Node
from exo.master.main import Master
from exo.routing.event_router import EventRouter
from exo.routing.router import Router, get_node_zid
from exo.shared.election import Election, ElectionResult
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    GlobalForwarderEvent,
    LocalForwarderEvent,
    NodeDownloadProgress,
    StateSnapshot,
)
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import DownloadPending
from exo.utils.channels import channel


class RecordingEventRouter(EventRouter):
    master_running_at_switch: list[bool]
    master_stopped: anyio.Event

    def switch_session(self, session_id: SessionId) -> None:
        self.master_running_at_switch.append(not self.master_stopped.is_set())
        super().switch_session(session_id)


async def test_a_demoted_master_stops_before_the_node_switches_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Catches the old master still producing events (which the event router would tag
    # with the new master's session) once the node has switched to the new session
    node_id = get_node_zid()
    own_session = SessionId(master_node_id=node_id, election_clock=1)
    new_session = SessionId(master_node_id=NodeId("other-master"), election_clock=2)
    event_router = RecordingEventRouter(
        own_session,
        command_sender=channel[ForwarderCommand]()[0],
        external_inbound=channel[GlobalForwarderEvent]()[1],
        external_outbound=channel[LocalForwarderEvent]()[0],
        snapshot_inbound=channel[StateSnapshot]()[1],
    )
    event_router.master_running_at_switch = []
    event_router.master_stopped = anyio.Event()
    master = Master(
        node_id,
        own_session,
        event_sender=event_router.sender(),
        global_event_sender=channel[GlobalForwarderEvent]()[0],
        snapshot_sender=channel[StateSnapshot]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        command_receiver=channel[ForwarderCommand]()[1],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
    )
    result_send, result_recv = channel[ElectionResult]()
    coordinator, _commands, download_events = _setup_coordinator(FakeShardDownloader())
    active_download = anyio.CancelScope()
    coordinator.active_downloads[MODEL_ID] = active_download
    status = DownloadPending(node_id=NODE_ID, shard_metadata=SHARD)
    coordinator.download_status[MODEL_ID] = status
    api = object.__new__(API)
    api.state = State()
    api.awaiting_snapshot = False
    api.paused_ev = anyio.Event()
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    snapshot_applied = anyio.Event()
    announce = coordinator.announce_to_master

    async def announce_and_receive_snapshot() -> None:
        await announce()
        # A fast new master can reply during the first awaited announcement.
        api._apply_snapshot(  # pyright: ignore[reportPrivateUsage]
            StateSnapshot(session=new_session, requester=SystemId(), state=State())
        )
        snapshot_applied.set()

    monkeypatch.setattr(
        coordinator, "announce_to_master", announce_and_receive_snapshot
    )
    node = Node(
        router=cast(Router, object()),
        event_router=event_router,
        download_coordinator=coordinator,
        worker=None,
        election=cast(Election, object()),
        election_result_receiver=result_recv,
        master=master,
        api=api,
        node_id=node_id,
        offline=False,
        _api_port=0,
    )

    async def run_master() -> None:
        await master.run()
        event_router.master_stopped.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(event_router.run)
        tg.start_soon(run_master)
        tg.start_soon(node._elect_loop)  # pyright: ignore[reportPrivateUsage]
        await anyio.sleep(0.05)

        await result_send.send(
            ElectionResult(session_id=new_session, won_clock=2, is_new_master=True)
        )
        with anyio.fail_after(5):
            await snapshot_applied.wait()

        assert event_router.master_running_at_switch == [False]
        assert node.master is None
        assert event_router.session_id == new_session
        assert node.download_coordinator is coordinator
        assert not active_download.cancel_called
        assert coordinator.active_downloads[MODEL_ID] is active_download
        assert not api.awaiting_snapshot
        announced = download_events.collect()
        assert len(announced) == 1
        assert isinstance(announced[0], NodeDownloadProgress)
        assert announced[0].download_progress == status
        result_send.close()
        tg.cancel_scope.cancel()
