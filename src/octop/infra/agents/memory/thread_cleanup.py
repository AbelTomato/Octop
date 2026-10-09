"""Delete LangGraph checkpoints without a live HarnessAgent.

Conversation text lives in the agent's memory store (SQLite file or
Postgres), not in Octop's ``threads`` table. Callers delete checkpoints
first and drop the thread row only after this succeeds, so a failure
leaves the conversation visible and retryable.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from octop.config import OctopConfig
from octop.infra.agents.memory.backend import open_memory_kwargs
from octop.infra.agents.workspace.dir import host_system_dir, workspace_dir_from_config_json
from octop.infra.errors import ErrorCode, OctopError

logger = logging.getLogger(__name__)

# ~8MB at the default 4KB page size. Matches
# octop_memory.pipeline.lifecycle.vacuum.DELETE_INCREMENTAL_VACUUM_PAGES.
_DELETE_VACUUM_PAGES = 2000


@dataclass(frozen=True)
class OrphanCheckpoint:
    agent_id: str
    thread_id: str
    nbytes: int


def workspace_for_agent_row(row: Any, *, paths: Any) -> Path:
    """On-disk workspace for an agent row. Does not create the directory."""
    return workspace_dir_from_config_json(
        getattr(row, "config_json", None),
        paths=paths,
        agent_id=str(row.agent_id),
        ensure=False,
    )


def delete_stored_thread(
    *,
    agent_id: str,
    thread_id: str,
    cfg: dict[str, Any],
    octop_config: OctopConfig,
    workspace_dir: Path,
) -> None:
    """Remove ``thread_id`` from the agent's checkpoint store.

    Returns normally when the store does not exist yet or the thread was
    never written. Raises ``CHECKPOINT_DELETE_FAILED`` when a store is
    present and the delete does not complete. A following SQLite reclaim
    is best-effort and never fails the delete.
    """
    ns, backend, backend_config, sqlite_path = _memory_location(
        agent_id=agent_id,
        cfg=cfg,
        octop_config=octop_config,
        workspace_dir=workspace_dir,
    )
    if backend == "sqlite" and (sqlite_path is None or not sqlite_path.is_file()):
        return

    memory = _open_memory(ns, backend, backend_config)
    try:
        try:
            memory.delete_thread(thread_id)
        except Exception as exc:
            if _store_has_no_checkpoints(exc):
                return
            raise _delete_failed(thread_id, exc) from exc
        _reclaim_sqlite(memory, agent_id=agent_id, thread_id=thread_id)
    finally:
        close_memory(memory)


def reclaim_stored_thread(
    *,
    agent_id: str,
    thread_id: str,
    cfg: dict[str, Any],
    octop_config: OctopConfig,
    workspace_dir: Path,
) -> None:
    """Best-effort SQLite reclaim after a live harness already deleted the rows."""
    ns, backend, backend_config, sqlite_path = _memory_location(
        agent_id=agent_id,
        cfg=cfg,
        octop_config=octop_config,
        workspace_dir=workspace_dir,
    )
    if backend != "sqlite" or sqlite_path is None or not sqlite_path.is_file():
        return
    memory = _open_memory(ns, backend, backend_config)
    try:
        _reclaim_sqlite(memory, agent_id=agent_id, thread_id=thread_id)
    finally:
        close_memory(memory)


def checkpoint_thread_bytes(
    *,
    agent_id: str,
    cfg: dict[str, Any],
    octop_config: OctopConfig,
    workspace_dir: Path,
) -> dict[str, int]:
    """``thread_id -> checkpoint payload bytes`` currently stored for the agent."""
    _ns, backend, backend_config, sqlite_path = _memory_location(
        agent_id=agent_id,
        cfg=cfg,
        octop_config=octop_config,
        workspace_dir=workspace_dir,
    )
    if backend == "sqlite":
        if sqlite_path is None or not sqlite_path.is_file():
            return {}
        return _sqlite_thread_bytes(sqlite_path)
    memory = _open_memory(_ns, backend, backend_config)
    try:
        return _postgres_thread_bytes(memory)
    finally:
        close_memory(memory)


def gc_orphan_checkpoints(
    services: Any, *, agent_id: str | None, apply: bool
) -> list[OrphanCheckpoint]:
    """Checkpoints whose thread row is already gone.

    ``apply=False`` only reports them. ``apply=True`` deletes each one
    through :func:`delete_stored_thread`.
    """
    rows = services.agent_repo.list_all(include_disabled=True)
    if agent_id is not None:
        rows = [row for row in rows if row.agent_id == agent_id]
    found: list[OrphanCheckpoint] = []
    for row in rows:
        cfg = agent_config_from_row(row)
        workspace = workspace_for_agent_row(row, paths=services.paths)
        usage = checkpoint_thread_bytes(
            agent_id=row.agent_id,
            cfg=cfg,
            octop_config=services.config,
            workspace_dir=workspace,
        )
        known = services.thread_repo.list_ids_for_agent(row.agent_id)
        orphans = [
            OrphanCheckpoint(agent_id=row.agent_id, thread_id=tid, nbytes=nbytes)
            for tid, nbytes in usage.items()
            if tid not in known
        ]
        if apply:
            for orphan in orphans:
                delete_stored_thread(
                    agent_id=row.agent_id,
                    thread_id=orphan.thread_id,
                    cfg=cfg,
                    octop_config=services.config,
                    workspace_dir=workspace,
                )
        found.extend(orphans)
    return found


def close_memory(memory: Any) -> None:
    """Release the checkpointer connection and the memory backend."""
    cp = getattr(memory, "_checkpointer", None)
    conn = getattr(cp, "conn", None)
    if conn is not None:
        try:
            conn.close()
        except Exception:
            logger.debug("checkpoint connection close failed", exc_info=True)
    pool = getattr(memory, "_checkpointer_pool", None)
    closer = getattr(pool, "close", None)
    if closer is not None:
        try:
            closer()
        except Exception:
            logger.debug("checkpoint pool close failed", exc_info=True)
    backend = getattr(memory, "_backend", None)
    backend_close = getattr(backend, "close", None)
    if backend_close is not None:
        try:
            backend_close()
        except Exception:
            logger.debug("memory backend close failed", exc_info=True)


def agent_config_from_row(row: Any) -> dict[str, Any]:
    raw = getattr(row, "config_json", None)
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _memory_location(
    *,
    agent_id: str,
    cfg: dict[str, Any],
    octop_config: OctopConfig,
    workspace_dir: Path,
) -> tuple[str, str, dict[str, Any] | None, Path | None]:
    ns, backend, backend_config = open_memory_kwargs(
        agent_id=agent_id,
        cfg=cfg,
        octop_config=octop_config,
        workspace_dir=workspace_dir,
    )
    sqlite_path: Path | None = None
    if backend == "sqlite":
        raw = (backend_config or {}).get("db_path")
        if raw:
            sqlite_path = Path(str(raw))
        else:
            sqlite_path = host_system_dir(workspace_dir, cfg) / "memory.sqlite"
    return ns, backend, backend_config if isinstance(backend_config, dict) else None, sqlite_path


def _open_memory(ns: str, backend: str, backend_config: dict[str, Any] | None) -> Any:
    from octop_memory.core import Memory

    return Memory(namespace=ns, backend=backend, backend_config=backend_config)


def _reclaim_sqlite(memory: Any, *, agent_id: str, thread_id: str) -> None:
    from octop_memory.pipeline.lifecycle.vacuum import nudge_vacuum
    from octop_memory.storage.backends.sqlite import SqliteMemoryBackend

    if not isinstance(getattr(memory, "_backend", None), SqliteMemoryBackend):
        return
    try:
        nudge_vacuum(memory, pages=_DELETE_VACUUM_PAGES)
    except Exception:
        logger.warning(
            "checkpoint reclaim failed agent=%s thread=%s",
            agent_id,
            thread_id,
            exc_info=True,
        )


def _sqlite_thread_bytes(path: Path) -> dict[str, int]:
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA query_only=ON")
        names = {
            str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "checkpoints" not in names:
            return {}
        rows = conn.execute(
            "SELECT thread_id, COALESCE(SUM(LENGTH(checkpoint)), 0) "
            "FROM checkpoints GROUP BY thread_id"
        ).fetchall()
    except sqlite3.Error as exc:
        if _store_has_no_checkpoints(exc):
            return {}
        raise
    finally:
        conn.close()
    return {str(thread_id): int(nbytes or 0) for thread_id, nbytes in rows}


def _postgres_thread_bytes(memory: Any) -> dict[str, int]:
    memory._ensure_checkpointer()
    saver = memory._checkpointer
    cursor = getattr(saver, "_cursor", None)
    if cursor is None:
        return {}
    try:
        with cursor() as cur:
            cur.execute(
                "SELECT thread_id, COALESCE(SUM(pg_column_size(checkpoint)), 0) AS nbytes "
                "FROM checkpoints GROUP BY thread_id"
            )
            rows = cur.fetchall()
    except Exception as exc:
        if _store_has_no_checkpoints(exc):
            return {}
        raise
    out: dict[str, int] = {}
    for row in rows:
        if isinstance(row, dict):
            out[str(row["thread_id"])] = int(row["nbytes"] or 0)
        else:
            out[str(row[0])] = int(row[1] or 0)
    return out


def _store_has_no_checkpoints(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "no such table" in text:
        return True
    return type(exc).__name__ == "UndefinedTable" or (
        "does not exist" in text and "checkpoint" in text
    )


def _delete_failed(thread_id: str, exc: BaseException) -> OctopError:
    logger.warning("checkpoint delete failed thread=%s", thread_id, exc_info=exc)
    return OctopError(
        ErrorCode.CHECKPOINT_DELETE_FAILED,
        f"could not delete conversation data for thread {thread_id!r}",
    )
