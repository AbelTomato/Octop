"""Checkpoint delete works without a running harness."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from octop.config import OctopConfig
from octop.infra.agents.manager import AgentManager
from octop.infra.agents.memory.thread_cleanup import (
    close_memory,
    delete_stored_thread,
    gc_orphan_checkpoints,
)
from octop.infra.db.migrate import run_migrations
from octop.infra.db.pool import SqlitePool
from octop.infra.db.services import build_shared_services
from octop.infra.errors import ErrorCode, OctopError
from octop.infra.users.manager import UserManager
from octop.infra.utils.paths import PathLayout


def _services(tmp_path: Path):
    paths = PathLayout(tmp_path / ".octop")
    paths.ensure_root()
    db = SqlitePool(paths.db)
    run_migrations(db)
    return build_shared_services(db=db, paths=paths, config=OctopConfig()), db


def _seed_checkpoint(path: Path, thread_id: str, payload: bytes = b"checkpoint-body") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from octop_memory.core import Memory

    memory = Memory(
        namespace="agent_seed",
        backend="sqlite",
        backend_config={"db_path": str(path)},
    )
    try:
        memory.delete_thread("warmup")
        conn = sqlite3.connect(path)
        conn.execute(
            "INSERT INTO checkpoints("
            "thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata"
            ") VALUES (?, '', 'cp1', 'msgpack', ?, ?)",
            (thread_id, payload, b"{}"),
        )
        conn.commit()
        conn.close()
    finally:
        close_memory(memory)


def _checkpoint_count(path: Path, thread_id: str) -> int:
    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
    finally:
        conn.close()
    return int(row[0])


def test_delete_stored_thread_removes_rows_when_agent_is_stopped(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    db_path = workspace / "memory.sqlite"
    _seed_checkpoint(db_path, "thr_gone")

    delete_stored_thread(
        agent_id="ag1",
        thread_id="thr_gone",
        cfg={},
        octop_config=OctopConfig(),
        workspace_dir=workspace,
    )

    assert _checkpoint_count(db_path, "thr_gone") == 0


def test_delete_stored_thread_missing_file_is_success(tmp_path: Path) -> None:
    delete_stored_thread(
        agent_id="ag1",
        thread_id="thr_new",
        cfg={},
        octop_config=OctopConfig(),
        workspace_dir=tmp_path / "missing",
    )


def test_delete_stored_thread_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    _seed_checkpoint(workspace / "memory.sqlite", "thr_keep")

    def _boom(self: object, thread_id: str) -> None:
        raise RuntimeError(f"locked {thread_id}")

    monkeypatch.setattr("octop_memory.core.Memory.delete_thread", _boom)

    with pytest.raises(OctopError) as exc:
        delete_stored_thread(
            agent_id="ag1",
            thread_id="thr_keep",
            cfg={},
            octop_config=OctopConfig(),
            workspace_dir=workspace,
        )
    assert exc.value.code == ErrorCode.CHECKPOINT_DELETE_FAILED
    assert _checkpoint_count(workspace / "memory.sqlite", "thr_keep") == 1


def test_reclaim_failure_still_deletes_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws"
    db_path = workspace / "memory.sqlite"
    _seed_checkpoint(db_path, "thr_vac")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("vacuum locked")

    monkeypatch.setattr("octop_memory.pipeline.lifecycle.vacuum.nudge_vacuum", _boom)

    delete_stored_thread(
        agent_id="ag1",
        thread_id="thr_vac",
        cfg={},
        octop_config=OctopConfig(),
        workspace_dir=workspace,
    )
    assert _checkpoint_count(db_path, "thr_vac") == 0


@pytest.mark.asyncio
async def test_manager_deletes_checkpoint_while_agent_is_stopped(tmp_path: Path) -> None:
    services, db = _services(tmp_path)
    try:
        user_id = services.user_repo.create(username="alice", password_hash="x", role="admin")
        services.agent_repo.create(agent_id="ag1", user_id=user_id, name="main")
        workspace = services.paths.agent_workspace("ag1")
        _seed_checkpoint(workspace / "memory.sqlite", "thr_1")
        manager = AgentManager(repos=services.repos, paths=services.paths, config=services.config)
        await manager.delete_thread_checkpoint("ag1", "thr_1")
        assert _checkpoint_count(workspace / "memory.sqlite", "thr_1") == 0
    finally:
        db.close()


def test_gc_orphans_reports_then_deletes(tmp_path: Path) -> None:
    services, db = _services(tmp_path)
    try:
        user_id = services.user_repo.create(username="alice", password_hash="x", role="admin")
        services.agent_repo.create(agent_id="ag1", user_id=user_id, name="main")
        services.thread_repo.insert(
            thread_id="thr_live",
            agent_id="ag1",
            user_id=user_id,
            channel_type="dashboard",
            session_key="dash",
        )
        workspace = services.paths.agent_workspace("ag1")
        db_path = workspace / "memory.sqlite"
        _seed_checkpoint(db_path, "thr_live", b"live")
        _seed_checkpoint(db_path, "thr_orphan", b"orphan-bytes")

        reported = gc_orphan_checkpoints(services, agent_id="ag1", apply=False)
        assert [item.thread_id for item in reported] == ["thr_orphan"]
        assert reported[0].nbytes == len(b"orphan-bytes")
        assert _checkpoint_count(db_path, "thr_orphan") == 1

        gc_orphan_checkpoints(services, agent_id="ag1", apply=True)
        assert _checkpoint_count(db_path, "thr_orphan") == 0
        assert _checkpoint_count(db_path, "thr_live") == 1
    finally:
        db.close()


def test_remove_user_deletes_workspace_and_foreign_checkpoints(tmp_path: Path) -> None:
    services, db = _services(tmp_path)
    try:
        alice = services.user_repo.create(username="alice", password_hash="x", role="admin")
        bob = services.user_repo.create(username="bob", password_hash="x", role="user")
        services.agent_repo.create(agent_id="ag_alice", user_id=alice, name="alice")
        services.agent_repo.create(agent_id="ag_bob", user_id=bob, name="bob")
        services.thread_repo.insert(
            thread_id="thr_bob_on_alice",
            agent_id="ag_alice",
            user_id=bob,
            channel_type="dashboard",
            session_key="bob-on-alice",
        )
        alice_ws = services.paths.agent_workspace("ag_alice")
        bob_ws = services.paths.agent_workspace("ag_bob")
        _seed_checkpoint(alice_ws / "memory.sqlite", "thr_bob_on_alice")
        bob_ws.mkdir(parents=True)
        (bob_ws / "notes.txt").write_text("keep-me-not", encoding="utf-8")
        services.paths.user_dir("bob").mkdir(parents=True)

        asyncio.run(UserManager(services).remove("bob"))

        assert services.user_repo.get_by_username("bob") is None
        assert services.agent_repo.get("ag_bob") is None
        assert not bob_ws.exists()
        assert not services.paths.user_dir("bob").exists()
        assert alice_ws.exists()
        assert _checkpoint_count(alice_ws / "memory.sqlite", "thr_bob_on_alice") == 0
        assert services.agent_repo.get("ag_alice") is not None
    finally:
        db.close()
