"""
TASK 3.3 - KNOWLEDGE BASE UPDATE SCHEDULER

The scheduled activation path for the knowledge base. It replaces an
earlier stub whose ``check_for_updates()`` returned ``False`` by
construction and whose "retry" only printed the 15/30/60 minute figure
without ever scheduling anything.

What a real run does:

1. Runs only inside the configured maintenance window, on a working day
   per the shared business calendar (so a Saturday or a configured
   holiday rolls to the next working day rather than activating).
2. Detects new and modified documents, and leaves unchanged ones alone.
3. Puts every candidate through **security screening** (prompt
   injection and unmasked sensitive data are rejected before anything
   else runs) and then the **quality gate** -- structural checks plus,
   for modifications of documents already serving, a grounding check
   against the serving corpus. A document that fails either is
   quarantined, with its reason, and never activated.
4. Activates, remembering the version it replaced, and **rebuilds the
   serving index** so the retrieval service actually serves what the
   metadata row says is active. A rebuild failure is an activation
   failure: rolled back and retried.
5. Watches the new version with a health check for
   ``health_check_window_minutes`` (5 by default) -- a genuine service
   probe of the serving index, not just the file check -- and
   **auto-rolls back to the previous version if a check fails**,
   rebuilding the index again to match.
6. Retries a failed activation at 15, then 30, then 60 minutes, and
   gives up after the last one instead of looping forever.

Every run is reported to the pipeline monitor (``log_run``, the default
monitor of the ``scheduler()`` entry point) once its outcome is
settled, with measured latency and confidence.

Everything is time-injected (``now=``) and dependency-injected
(``PipelinePorts``), so the whole schedule -- windows, retries, the
rollback deadline -- is exercisable without sleeping, without a real
database and without touching the live knowledge base. The test suite
drives simulated time through these same entry points.

All public functions take and return timezone-aware UTC datetimes.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


# --------------------------------------------------
# shared business calendar
# --------------------------------------------------

# knowledge_pipeline/ and src/ are sibling packages and neither is
# installed. Import the calendar by name when src/ is already on the
# path (the normal case when this module is imported from src/) and
# fall back to adding it explicitly, so the module works from either
# working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = PROJECT_ROOT / "src"

try:

    from business_calendar import (  # noqa: F401
        BusinessCalendarConfig,
        as_utc,
        is_working_day,
        local_now,
        parse_hhmm,
    )

except ImportError:  # pragma: no cover - depends on invocation cwd

    if str(_SRC) not in sys.path:

        sys.path.insert(0, str(_SRC))

    from business_calendar import (  # noqa: F401
        BusinessCalendarConfig,
        as_utc,
        is_working_day,
        local_now,
        parse_hhmm,
    )


MINUTES_PER_DAY = 24 * 60

# Day-by-day loop guard, matching business_calendar's safety valve.
MAX_DAYS = 3660


# --------------------------------------------------
# run statuses
# --------------------------------------------------

STATUS_SKIPPED = "skipped"
STATUS_NO_CHANGES = "no_changes"
STATUS_ACTIVATED = "activated"
STATUS_QUARANTINED = "quarantined"
STATUS_ROLLED_BACK = "rolled_back"
STATUS_PENDING_RETRY = "pending_retry"
STATUS_FAILED = "failed"

# Statuses whose outcome is settled the moment activation returns.
# Deliberately excludes "activated": an activation is not confirmed
# until its health-check window has elapsed, so it is reported from
# health_check() instead. That keeps the monitor at exactly one event
# per run, whatever the run did.
FINAL_STATUSES = (
    STATUS_NO_CHANGES,
    STATUS_QUARANTINED,
    STATUS_FAILED,
)


# --------------------------------------------------
# configuration
# --------------------------------------------------

@dataclass
class SchedulerConfig:
    """
    The update schedule.

    Mutable at runtime like every other config in this project: the
    evaluation may move the update time, widen the maintenance window
    or shorten the health-check deadline while the process is running,
    and the next ``activate()`` honours the new values because no
    window or retry instant is cached.
    """

    # Local wall-clock start of the maintenance window, "HH:MM".
    update_time: str = "02:00"

    # How long the window stays open. May wrap past midnight, so
    # "23:30" + 60 minutes means "until 00:30". The default is wide
    # enough to contain the whole retry ladder below: a failure one
    # minute into the window must still have room for its 60-minute
    # retry, or the last retry would always fall outside the window.
    maintenance_window_minutes: int = 120

    # Retry delays after a failed activation, in minutes.
    retry_minutes: Tuple[int, ...] = (15, 30, 60)

    # Whether a retry must also land inside a maintenance window.
    # Default True, because the window exists precisely to stop the
    # knowledge base changing outside it -- a retry is still an
    # activation. A retry that comes due after the window has closed
    # waits for the next one instead of running anyway.
    retry_requires_window: bool = True

    # The post-activation health-check deadline. A failure inside this
    # window triggers an automatic rollback.
    health_check_window_minutes: int = 5
    health_check_interval_minutes: int = 1

    quality_gate_enabled: bool = True

    # When False, candidates skip the security screening port even if
    # one is wired (the port being None already skips it). A screen
    # that finds injection content or unmasked sensitive data still
    # rejects and quarantines the document regardless of this flag.
    security_screen_enabled: bool = True

    # Whether activation must also rebuild the serving index. Disabling
    # it is an operator escape hatch for a deployment whose index is
    # managed elsewhere; the default keeps the metadata row and the
    # retrieval service from drifting apart.
    rebuild_index_enabled: bool = True

    # When True, a window that falls on a weekend or holiday is skipped
    # and the run waits for the next working day.
    skip_non_working_days: bool = True

    calendar: BusinessCalendarConfig = field(
        default_factory=BusinessCalendarConfig
    )

    def __post_init__(self) -> None:
        """
        Validate loudly.

        A typo here would silently disable the update path, which is
        exactly the failure mode the previous stub had. Better to raise
        at construction.
        """

        # Raises ValueError on a malformed "HH:MM".
        parse_hhmm(self.update_time)

        if self.maintenance_window_minutes <= 0:

            raise ValueError(
                "maintenance_window_minutes must be positive"
            )

        if self.health_check_window_minutes <= 0:

            raise ValueError(
                "health_check_window_minutes must be positive"
            )

        if self.health_check_interval_minutes <= 0:

            raise ValueError(
                "health_check_interval_minutes must be positive"
            )

        if not self.retry_minutes:

            raise ValueError("retry_minutes must not be empty")

        if any(delay <= 0 for delay in self.retry_minutes):

            raise ValueError("retry_minutes must all be positive")

        if list(self.retry_minutes) != sorted(self.retry_minutes):

            raise ValueError(
                "retry_minutes must be in ascending order"
            )

    # ----------------------------------------------

    def window_start_minute(self) -> int:
        """Minutes past local midnight at which the window opens."""

        start = parse_hhmm(self.update_time)

        return start.hour * 60 + start.minute

    def retry_delays(self) -> Tuple[int, ...]:
        """
        The configured delays, always as a tuple.

        Safe to hand out: a caller cannot mutate the config through an
        immutable sequence, only replace the attribute. Normalises a
        list assigned at runtime so the rest of the scheduler can index
        it without caring what type the config was given.
        """

        return tuple(self.retry_minutes)

    def max_attempts(self) -> int:
        """The original run plus one attempt per retry delay."""

        return len(self.retry_minutes) + 1


# --------------------------------------------------
# window arithmetic
# --------------------------------------------------

def _minute_of_day(moment_local: datetime) -> int:
    """Minutes past local midnight for an aware local datetime."""

    return moment_local.hour * 60 + moment_local.minute


def is_maintenance_window(
    now: Optional[datetime],
    config: Optional[SchedulerConfig] = None,
) -> bool:
    """
    Whether ``now`` falls inside an open maintenance window.

    Half-open: the start instant is inside, the end instant is not, so
    a 60-minute window from 02:00 covers 02:00:00 through 02:59:59 and
    a run cannot be counted in two windows.
    """

    config = config or SchedulerConfig()

    local = local_now(now, config.calendar)

    if config.skip_non_working_days and not is_working_day(
        local.date(), config.calendar
    ):

        return False

    elapsed = (
        _minute_of_day(local) - config.window_start_minute()
    ) % MINUTES_PER_DAY

    return elapsed < config.maintenance_window_minutes


def next_maintenance_start(
    now: Optional[datetime],
    config: Optional[SchedulerConfig] = None,
) -> datetime:
    """
    The next window opening strictly after ``now``, in UTC.

    Strictly after, because the caller is asking when an update that
    did *not* run now will next get its chance. A run that happens to
    be sitting exactly on the boundary is already covered by
    ``is_maintenance_window``.
    """

    config = config or SchedulerConfig()

    local = local_now(now, config.calendar)

    zone = local.tzinfo

    for offset in range(MAX_DAYS):

        day = local.date() + timedelta(days=offset)

        if config.skip_non_working_days and not is_working_day(
            day, config.calendar
        ):

            continue

        # Attaching the zone to the wall-clock time lets ZoneInfo
        # resolve that day's own UTC offset, so a window on the far
        # side of a DST change still opens at the local time asked for.
        candidate = datetime(
            day.year,
            day.month,
            day.day,
            tzinfo=zone,
        ) + timedelta(minutes=config.window_start_minute())

        if candidate > local:

            return candidate.astimezone(timezone.utc)

    raise ValueError(
        "No maintenance window found within "
        f"{MAX_DAYS} days; check the calendar configuration."
    )


def window_closes_at(
    now: Optional[datetime],
    config: Optional[SchedulerConfig] = None,
) -> Optional[datetime]:
    """
    When the window containing ``now`` closes, or None if it is shut.

    Used to report the deadline for the current run rather than to
    enforce it: retries may legitimately run past it.
    """

    config = config or SchedulerConfig()

    if not is_maintenance_window(now, config):

        return None

    local = local_now(now, config.calendar)

    elapsed = (
        _minute_of_day(local) - config.window_start_minute()
    ) % MINUTES_PER_DAY

    remaining = config.maintenance_window_minutes - elapsed

    return as_utc(now) + timedelta(minutes=remaining)


# --------------------------------------------------
# retry arithmetic
# --------------------------------------------------

def retry_schedule(
    failed_at: Optional[datetime],
    config: Optional[SchedulerConfig] = None,
) -> List[datetime]:
    """
    The UTC instants at which a failure at ``failed_at`` is retried.

    Plain wall-clock offsets, not business minutes: a retry is a short
    operational backoff measured from the failure, not an SLA with a
    working-hours clock.
    """

    config = config or SchedulerConfig()

    moment = as_utc(failed_at)

    return [
        moment + timedelta(minutes=delay)
        for delay in config.retry_delays()
    ]


def next_retry_at(
    failed_at: Optional[datetime],
    attempt_index: int,
    config: Optional[SchedulerConfig] = None,
) -> Optional[datetime]:
    """
    When attempt ``attempt_index`` should run, or None once exhausted.

    ``attempt_index`` is the index of the attempt that already
    happened, so 0 (the original run) yields the first retry delay.
    """

    config = config or SchedulerConfig()

    delays = config.retry_delays()

    if attempt_index < 0 or attempt_index >= len(delays):

        return None

    return as_utc(failed_at) + timedelta(minutes=delays[attempt_index])


# --------------------------------------------------
# records
# --------------------------------------------------

@dataclass
class HealthCheck:
    """One post-activation probe."""

    checked_at: datetime
    minutes_after_activation: float
    healthy: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, object]:

        return {
            "checked_at": self.checked_at.isoformat(),
            "minutes_after_activation": round(
                self.minutes_after_activation, 4
            ),
            "healthy": self.healthy,
            "detail": self.detail,
        }


@dataclass
class UpdateAttempt:
    """
    One try at activating the change set.

    ``index`` 0 is the scheduled run; 1.. are the retries.
    """

    index: int
    attempted_at: datetime
    outcome: str
    detail: str = ""
    retry_after_minutes: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:

        return {
            "index": self.index,
            "attempted_at": self.attempted_at.isoformat(),
            "outcome": self.outcome,
            "detail": self.detail,
            "retry_after_minutes": self.retry_after_minutes,
        }


@dataclass
class RunRecord:
    """
    A whole update run, carried across ticks.

    ``activate()`` returns one of these and ``health_check()`` takes it
    back, because monitoring happens after activation and therefore
    cannot live inside a single call. That is also what makes the
    five-minute rollback deadline testable: the record knows exactly
    when activation happened.
    """

    started_at: datetime
    status: str
    finished_at: Optional[datetime] = None

    detected: Dict[str, List[str]] = field(default_factory=dict)
    skipped_unchanged: List[str] = field(default_factory=list)

    activated: List[str] = field(default_factory=list)
    versions: Dict[str, int] = field(default_factory=dict)
    previous_versions: Dict[str, int] = field(default_factory=dict)

    quarantined: List[str] = field(default_factory=list)
    quarantine_reasons: Dict[str, str] = field(default_factory=dict)

    rolled_back: List[str] = field(default_factory=list)
    rollback_reason: str = ""
    rollback_minutes: Optional[float] = None

    attempts: List[UpdateAttempt] = field(default_factory=list)
    health_checks: List[HealthCheck] = field(default_factory=list)

    next_retry_at: Optional[datetime] = None
    next_window_at: Optional[datetime] = None
    monitoring_complete: bool = False

    notes: List[str] = field(default_factory=list)

    # Real wall-clock seconds the activation attempt took, measured by
    # the scheduler around ``_activate``. The run's *scheduled* instants
    # stay on started_at/finished_at (simulated in tests); this is the
    # measured latency the monitor logs. None until the attempt runs.
    latency_seconds: Optional[float] = None

    # Deliberately unannotated, so it stays a plain class attribute
    # rather than becoming a dataclass field with an __init__ keyword.
    # The scheduler sets it from its live config, so the deadline is
    # always checked against the current policy.
    _health_window_minutes = 5

    # ----------------------------------------------

    @property
    def attempted(self) -> int:
        """How many activations have been tried."""

        return len(self.attempts)

    @property
    def exhausted(self) -> bool:
        """No retries left and nothing activated."""

        return (
            self.status == STATUS_FAILED
            and self.next_retry_at is None
        )

    @property
    def rolled_back_within_window(self) -> bool:
        """
        Whether the automatic rollback met its own deadline.

        False when no rollback happened, so a test cannot pass by
        asserting a deadline on a run that never rolled back.
        """

        if self.status != STATUS_ROLLED_BACK:

            return False

        if self.rollback_minutes is None:

            return False

        return self.rollback_minutes <= self._health_window_minutes

    def to_dict(self) -> Dict[str, object]:

        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": (
                self.finished_at.isoformat()
                if self.finished_at
                else None
            ),
            "status": self.status,
            "detected": self.detected,
            "skipped_unchanged": list(self.skipped_unchanged),
            "activated": list(self.activated),
            "versions": dict(self.versions),
            "previous_versions": dict(self.previous_versions),
            "quarantined": list(self.quarantined),
            "quarantine_reasons": dict(self.quarantine_reasons),
            "rolled_back": list(self.rolled_back),
            "rollback_reason": self.rollback_reason,
            "rollback_minutes": self.rollback_minutes,
            "attempts": [a.to_dict() for a in self.attempts],
            "health_checks": [
                h.to_dict() for h in self.health_checks
            ],
            "next_retry_at": (
                self.next_retry_at.isoformat()
                if self.next_retry_at
                else None
            ),
            "monitoring_complete": self.monitoring_complete,
            "notes": list(self.notes),
            "latency_seconds": self.latency_seconds,
        }


# --------------------------------------------------
# injected dependencies
# --------------------------------------------------

@dataclass
class PipelinePorts:
    """
    The pipeline this scheduler drives, as callables.

    Injected rather than imported so the scheduler can be exercised
    against a fake knowledge base. The real wiring -- duplicate
    detection, the quality validator, version_manager and rollback --
    is assembled by ``default_ports()`` and is the only thing that
    touches the filesystem or metadata.db.

    ``activate`` returns the new version number, and ``rollback``
    takes the version to restore.

    Two later additions, both optional so every existing construction
    keeps working:

    ``rebuild_index`` rebuilds the serving index once activation has
    succeeded. It defaults to a no-op (None) so a fake pipeline never
    touches a vector store; ``default_ports()`` supplies the real one.

    ``security_screen`` screens a candidate before anything else runs:
    prompt-injection content and unmasked sensitive data are rejected
    (and quarantined) rather than activated. Also None by default.
    """

    detect: Callable[[], Dict[str, List[str]]]
    activate: Callable[[str], int]
    current_version: Callable[[str], int]
    rollback: Callable[[str, int], None]
    health_check: Callable[[str], Tuple[bool, str]]
    quality_gate: Optional[Callable[[str], Tuple[bool, str]]] = None
    quarantine: Optional[Callable[[str, str], None]] = None

    # Called with no arguments after a successful activation, to make
    # what was activated actually servable.
    rebuild_index: Optional[Callable[[], None]] = None

    # (filename) -> (allowed, reason); runs before the quality gate.
    security_screen: Optional[Callable[[str], Tuple[bool, str]]] = None


def _load_module(name: str, path: Path):
    """
    Load a module from an explicit file path.

    By path rather than by name because the pieces of this pipeline are
    split across two directories that are neither installed nor each
    other's siblings: the change detector lives in knowledge_base/ and
    the versioning and rollback code lives here. A bare ``import
    update_pipeline`` works only when the caller happens to have
    knowledge_base/ on sys.path, which is exactly the fragile
    assumption that made the previous wiring fail.
    """

    import importlib.util

    if not path.is_file():

        raise ImportError(f"Pipeline module is missing: {path}")

    spec = importlib.util.spec_from_file_location(name, path)

    if spec is None or spec.loader is None:

        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# --------------------------------------------------
# the serving index
# --------------------------------------------------

# Activation used to write a metadata row and stop there: the FAISS
# index the retrieval service actually reads was never touched, so an
# "activated" document was invisible to customers until some other
# process happened to rebuild the index. The two functions below give
# activation a real index rebuild and the health check a real service
# probe, without making faiss or langchain a requirement: everything
# vector-related is imported lazily and only when an embedding
# function has been injected.

# The dependency-free half of the serving index: a manifest of the
# documents the index was built from, with their content and hashes,
# written atomically next to the vector files.
INDEX_MANIFEST_NAME = "index_manifest.json"

# A probe query for callers without a document at hand; deliberately
# this support bot's vocabulary.
DEFAULT_PROBE_QUERY = "refund policy shipping support order"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _serving_documents(documents_dir: Path) -> List[Path]:
    """The active documents: the files in the documents directory."""

    return sorted(
        (
            path
            for path in documents_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        ),
        key=lambda path: path.name,
    )


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while chunk := handle.read(4096):
            digest.update(chunk)

    return digest.hexdigest()


def _read_text(path: Path) -> str:
    """Read a document; a binary document is not servable text."""

    return path.read_text(encoding="utf-8")


def _probe_tokens(query: Union[str, Sequence[str]]) -> List[str]:
    """
    The meaningful tokens of a probe query.

    Word runs of 3+ characters (lightly de-pluralised) and numbers of
    2+ digits, so "refunds within 30 days" probes for "refund", "30".
    """

    if isinstance(query, str):
        tokens = _TOKEN_PATTERN.findall(query.lower())
    else:
        tokens = [
            token.lower()
            for token in query
        ]

    meaningful = []

    for token in tokens:

        if token.isdigit():

            if len(token) >= 2:
                meaningful.append(token)

        elif len(token) >= 3:

            term = token[:-1] if token.endswith("s") and len(token) > 3 else token

            meaningful.append(term)

    return meaningful


def _lexical_retrieve(
    query: Union[str, Sequence[str]],
    entries: List[Dict[str, str]],
    k: int = 3,
) -> List[Dict[str, object]]:
    """
    Deterministic lexical retrieval over manifest entries.

    ``entries`` are dicts with ``filename`` and ``content``. Scores by
    the number of probe tokens contained in each document's text and
    returns the top ``k`` with score > 0, best first, ties by name.
    """

    tokens = _probe_tokens(query)

    if not tokens:
        return []

    scored = []

    for entry in entries:

        content = entry.get("content", "").lower()

        score = sum(1 for token in tokens if token in content)

        if score > 0:
            scored.append({"filename": entry["filename"], "score": score})

    scored.sort(key=lambda item: (-item["score"], item["filename"]))

    return scored[:k]


def _build_faiss_index(
    documents: List[Path],
    index_dir: Path,
    embed_fn: Callable[[str], List[float]],
) -> None:
    """
    Build a real FAISS index over the documents, in the format
    ``FAISS.load_local`` reads.

    Imports the vector stack *here and nowhere else*: this function is
    only reached when an embedding function was injected, and the
    ImportError it raises in an environment without faiss/langchain is
    exactly the signal to fall back to the manifest index.
    """

    from langchain_community.vectorstores import FAISS  # noqa: PLC0415
    from langchain_core.documents import Document  # noqa: PLC0415

    class _InjectedEmbeddings:
        """Adapt ``embed_fn`` to langchain's Embeddings interface."""

        def __init__(self, fn: Callable[[str], List[float]]):
            self._fn = fn

        def embed_documents(self, texts):
            return [self._fn(text) for text in texts]

        def embed_query(self, text):
            return self._fn(text)

    pages = [
        Document(
            page_content=_read_text(path),
            metadata={"source": path.name, "filename": path.name},
        )
        for path in documents
    ]

    store = FAISS.from_documents(pages, _InjectedEmbeddings(embed_fn))

    index_dir.mkdir(parents=True, exist_ok=True)

    store.save_local(str(index_dir))


def rebuild_serving_index(
    documents_dir: Optional[Path] = None,
    index_dir: Optional[Path] = None,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
) -> Dict[str, object]:
    """
    Rebuild the serving index from the active documents.

    Call this from the app after the knowledge base changes (the
    scheduler already calls it via its ``rebuild_index`` port on every
    activation)::

        from scheduler import rebuild_serving_index
        rebuild_serving_index(embed_fn=my_embeddings.embed_query)

    ``embed_fn`` maps text to a vector. With it *and* faiss/langchain
    importable, a real FAISS index (``index.faiss`` + ``index.pkl``,
    loadable with ``FAISS.load_local``) is written over the documents.
    Without either, the build degrades to the dependency-free manifest
    index -- a snapshot of the documents' content and hashes that
    ``probe_serving_index`` serves and verifies against -- and says so
    in the returned summary; it never requires the vector stack to be
    installed.

    Returns ``{"engine", "documents", "index_dir", "notes"}``.
    """

    documents_dir = Path(
        documents_dir
        or (PROJECT_ROOT / "knowledge_base" / "documents")
    )

    index_dir = Path(
        index_dir
        or (documents_dir.parent / "faiss_index")
    )

    if not documents_dir.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {documents_dir}"
        )

    documents = _serving_documents(documents_dir)

    entries = []

    for path in documents:

        content = _read_text(path)

        entries.append(
            {
                "filename": path.name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "content": content,
            }
        )

    engine = "manifest"
    notes: List[str] = []

    if embed_fn is not None:

        try:
            _build_faiss_index(documents, index_dir, embed_fn)
            engine = "faiss"

        except ImportError as error:
            notes.append(
                "vector stack unavailable "
                f"({error}); wrote the dependency-free manifest index"
            )

        except Exception as error:  # noqa: BLE001
            notes.append(
                "FAISS build failed "
                f"({error}); wrote the dependency-free manifest index"
            )

    manifest = {
        "engine": engine,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "embedder": (
            getattr(embed_fn, "__name__", "custom")
            if embed_fn is not None
            else None
        ),
        "documents": entries,
        "notes": notes,
    }

    index_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = index_dir / INDEX_MANIFEST_NAME

    # Atomic write: a rebuild must never leave a half-written manifest
    # behind for the health probe to trip over.
    temporary = manifest_path.with_name(INDEX_MANIFEST_NAME + ".tmp")

    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    temporary.replace(manifest_path)

    print(
        f"🧭 Serving index rebuilt ({engine}): "
        f"{len(entries)} document(s) -> {manifest_path}"
    )

    return {
        "engine": engine,
        "documents": len(entries),
        "index_dir": str(index_dir),
        "notes": notes,
    }


def probe_serving_index(
    index_dir: Optional[Path] = None,
    documents_dir: Optional[Path] = None,
    probe_query: Optional[Union[str, Sequence[str]]] = None,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    k: int = 3,
) -> Tuple[bool, str]:
    """
    A genuine service probe: the serving index must exist, load and
    answer a query with non-empty evidence.

    What it checks, in order:

    1. The index directory exists -- no index, no service.
    2. The manifest written by ``rebuild_serving_index`` loads, lists
       at least one document, and (when ``documents_dir`` is given)
       still matches the documents on disk: a document that changed or
       vanished since the rebuild is a stale index.
    3. Retrieval over the indexed documents for ``probe_query``
       (default: this bot's vocabulary) returns evidence.
    4. For a FAISS-engine index, when ``embed_fn`` and faiss are both
       available: the vector index loads and answers the query too.

    Returns ``(healthy, detail)``; the detail explains exactly what was
    probed so a caller can tell "never deployed" from "broken".
    """

    documents_dir = (
        Path(documents_dir)
        if documents_dir is not None
        else None
    )

    if index_dir is None:

        index_dir = (
            documents_dir.parent / "faiss_index"
            if documents_dir is not None
            else PROJECT_ROOT / "knowledge_base" / "faiss_index"
        )

    else:
        index_dir = Path(index_dir)

    if not index_dir.is_dir():
        return False, f"serving index not found at {index_dir}"

    manifest_path = index_dir / INDEX_MANIFEST_NAME

    if not manifest_path.is_file():

        # An index built by another tool (ingest.py's FAISS store, for
        # instance) has no manifest. It is still the service this
        # deployment serves from, so verify its files exist and are not
        # empty rather than declaring a working service down.
        artifacts = [
            index_dir / "index.faiss",
            index_dir / "index.pkl",
        ]

        if all(path.is_file() and path.stat().st_size > 0 for path in artifacts):
            return (
                True,
                "serving index present "
                "(structural check; no manifest from this pipeline)",
            )

        return False, "serving index files are missing or empty"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    except (OSError, ValueError) as error:
        return False, f"serving index manifest is unreadable: {error}"

    entries = list(manifest.get("documents") or [])

    if not entries:
        return False, "serving index is empty (no documents indexed)"

    if documents_dir is not None:

        for entry in entries:

            live = documents_dir / entry["filename"]

            if not live.is_file():
                return (
                    False,
                    "stale serving index: "
                    f"{entry['filename']} is no longer in the documents "
                    "directory",
                )

            if _sha256_file(live) != entry.get("sha256"):
                return (
                    False,
                    "stale serving index: "
                    f"{entry['filename']} changed since the index was built",
                )

    query = probe_query if probe_query is not None else DEFAULT_PROBE_QUERY

    evidence = _lexical_retrieve(query, entries, k=k)

    if not evidence:
        return (
            False,
            "probe returned no evidence "
            f"(query: {query!r})",
        )

    detail = (
        "evidence: "
        + ", ".join(
            f"{item['filename']} (score {item['score']})"
            for item in evidence
        )
    )

    # The vector-level check when this index is a FAISS one and the
    # caller can embed. If faiss is simply not installed in *this*
    # interpreter (the scheduler running under plain python3 while the
    # app serves with the venv), that is not the service being down;
    # the lexical evidence above already proved the documents serve.
    if manifest.get("engine") == "faiss" and embed_fn is not None:

        try:
            import faiss  # noqa: PLC0415

            vector_index = faiss.read_index(
                str(index_dir / "index.faiss")
            )

            if vector_index.ntotal <= 0:
                return False, "serving vector index is empty"

            import numpy  # noqa: PLC0415  (faiss needs it anyway)

            vector = numpy.array(
                [embed_fn(query if isinstance(query, str) else " ".join(query))],
                dtype="float32",
            )

            distances, ids = vector_index.search(vector, k)

            if len(ids[0]) == 0:
                return False, "serving vector index returned no results"

            detail = f"vector probe answered; {detail}"

        except ImportError:
            detail = f"vector probe skipped (faiss not installed); {detail}"

        except Exception as error:  # noqa: BLE001
            return False, f"serving vector index failed to load: {error}"

    return True, detail


def default_ports(
    documents_dir: Optional[Path] = None,
    metadata_db: Optional[Path] = None,
    index_dir: Optional[Path] = None,
    quarantine_dir: Optional[Path] = None,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
) -> PipelinePorts:
    """
    The real pipeline, wired to the sibling modules.

    Modules are resolved by path and the metadata database is passed
    explicitly, so the whole wiring is independent of the current
    working directory. That matters: ``version_manager`` and
    ``rollback`` default to the relative ``"metadata.db"``, so a
    scheduler started from anywhere but the knowledge_pipeline
    directory would have activated versions in a second, empty
    database and reported success.

    ``index_dir`` (default ``<documents_dir>/../faiss_index``) and
    ``quarantine_dir`` (default ``<documents_dir>/../quarantine``) are
    likewise resolved next to the documents they describe, so a wiring
    given a scratch documents directory never writes into the real
    knowledge base.

    ``embed_fn`` (text -> vector) is what makes the index real: with
    it, ``rebuild_index`` builds an actual FAISS index; without it (or
    without faiss/langchain installed) the rebuild degrades to the
    dependency-free manifest index. The scheduler's own tests run
    entirely without the vector stack.
    """

    import sqlite3

    pipeline_dir = Path(__file__).resolve().parent

    documents_dir = Path(
        documents_dir
        or (PROJECT_ROOT / "knowledge_base" / "documents")
    )
    metadata_db = metadata_db or (PROJECT_ROOT / "metadata.db")
    index_dir = Path(
        index_dir
        or (documents_dir.parent / "faiss_index")
    )
    quarantine_dir = Path(
        quarantine_dir
        or (documents_dir.parent / "quarantine")
    )

    quality_validator = _load_module(
        "kp_quality_validator", pipeline_dir / "quality_validator.py"
    )
    version_manager = _load_module(
        "kp_version_manager", pipeline_dir / "version_manager.py"
    )
    rollback_module = _load_module(
        "kp_rollback", pipeline_dir / "rollback.py"
    )
    update_pipeline = _load_module(
        "kp_update_pipeline",
        PROJECT_ROOT / "knowledge_base" / "update_pipeline.py",
    )

    # Security helpers, loaded the same way: by path, so knowledge_base
    # never has to be on sys.path and those files are used, not copied.
    security = _load_module(
        "kp_security",
        PROJECT_ROOT / "knowledge_base" / "security.py",
    )
    pii_masker = _load_module(
        "kp_pii_masker",
        PROJECT_ROOT / "knowledge_base" / "multimodal" / "pii_masker.py",
    )

    def _connection():

        conn = sqlite3.connect(metadata_db)
        conn.row_factory = sqlite3.Row
        return conn

    def detect() -> Dict[str, List[str]]:

        if not Path(documents_dir).is_dir():

            return {"new": [], "modified": [], "unchanged": []}

        return update_pipeline.detect_changes(documents_dir)

    def current_version(filename: str) -> int:

        conn = _connection()

        try:

            row = conn.execute(
                "SELECT MAX(version) AS version FROM documents "
                "WHERE filename=?",
                (filename,),
            ).fetchone()

        finally:

            conn.close()

        if row is None or row["version"] is None:

            return 0

        return int(row["version"])

    def activate(filename: str) -> int:

        path = Path(documents_dir) / filename

        file_hash = update_pipeline.calculate_file_hash(str(path))

        version_manager.save_version(
            filename, file_hash, db_path=metadata_db
        )

        return current_version(filename)

    def rollback(filename: str, target_version: int) -> None:

        rollback_module.rollback(
            filename, target_version, db_path=metadata_db
        )

    def health_check(filename: str) -> Tuple[bool, str]:
        """
        The document must still be there and still pass the quality
        gate -- and the retrieval service must be able to serve
        evidence for it. The gate alone was the old (only) probe; a
        document that passes its file check while the serving index is
        missing, stale or silent is not a healthy activation.
        """

        path = Path(documents_dir) / filename

        if not path.is_file():

            return False, "document missing after activation"

        passed, message = quality_validator.validate_document(path)

        if not passed:

            return False, message

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return False, "document is no longer valid UTF-8 text"

        # Probe the service with the document's own key terms: after
        # activating X, the serving index must answer a query about
        # X's subject matter with evidence.
        probe_query = " ".join(
            quality_validator.extract_key_terms(content)[:8]
        )

        healthy, detail = probe_serving_index(
            index_dir=index_dir,
            documents_dir=documents_dir,
            probe_query=probe_query,
            embed_fn=embed_fn,
        )

        if healthy:

            return True, f"ok (quality gate passed; serving index: {detail})"

        if detail.startswith("serving index not found"):

            # No index has ever been built for this documents
            # directory. Rather than calling a service that was never
            # deployed "down", probe retrieval over the active
            # documents themselves -- still a real check that the
            # document is retrievable, not just present on disk.
            entries = [
                {
                    "filename": doc.name,
                    "content": doc.read_text(encoding="utf-8"),
                }
                for doc in _serving_documents(documents_dir)
            ]

            evidence = _lexical_retrieve(probe_query, entries)

            if evidence:
                return (
                    True,
                    "ok (quality gate passed; no serving index built "
                    "yet -- probed the active documents directly)",
                )

            return (
                False,
                f"service probe returned no evidence for {filename}",
            )

        return False, f"serving index unhealthy: {detail}"

    def quality_gate(filename: str) -> Tuple[bool, str]:
        """
        The quality gate, with the grounding stage wired in.

        A brand-new document is checked structurally only: new topics
        are the point of a new document. A *modification* of a
        document that is already serving is held to the grounding gate
        against the rest of the serving corpus, so an update whose
        content the rest of the knowledge base does not support -- the
        "reduces accuracy or grounding" case -- is rejected instead of
        activated.
        """

        # The ports contract passes a filename; validate_document takes
        # a path. Resolving it here is what keeps the gate pointed at
        # the documents directory rather than at the working directory,
        # which is where a bare filename would otherwise land.
        path = Path(documents_dir) / filename

        if current_version(filename) > 0 and documents_dir.is_dir():

            corpus = [
                doc
                for doc in _serving_documents(documents_dir)
                if doc.name != filename
            ]

            return quality_validator.validate_document(
                path, corpus=corpus
            )

        return quality_validator.validate_document(path)

    def quarantine(filename: str, reason: str) -> None:
        """
        Move the offending document out of the documents directory,
        for real, and record why.

        The old hook only printed, which left a document that had just
        failed a gate sitting in the serving directory where the next
        run would detect and process it again. It now lands in
        ``quarantine_dir`` (created if needed) with the reason in a
        sidecar ``<filename>.reason`` file, and is never left in place.
        """

        quarantine_dir.mkdir(parents=True, exist_ok=True)

        source = Path(documents_dir) / filename
        destination = quarantine_dir / filename

        if source.is_file():
            shutil.move(str(source), str(destination))

        (quarantine_dir / f"{filename}.reason").write_text(
            f"{reason}\n", encoding="utf-8"
        )

        print(f"🚧 Quarantined {filename}: {reason}")
        print(f"Moved to: {destination}")

    def rebuild_index() -> None:
        """Make what was activated actually servable."""

        rebuild_serving_index(
            documents_dir=documents_dir,
            index_dir=index_dir,
            embed_fn=embed_fn,
        )

    def security_screen(filename: str) -> Tuple[bool, str]:
        """
        Screen a candidate before anything else touches it.

        Two rejections, both using the project's existing helpers
        (loaded by path, never edited): documents containing
        prompt-injection instructions, and documents containing
        *unmasked* sensitive data. Masking is deliberately not done
        in place here -- a knowledge document that arrives carrying
        raw customer PII is a submission problem, and quietly
        rewriting the submitter's file would hide it. The document is
        rejected with the reason; the masked version can be submitted
        again.
        """

        path = Path(documents_dir) / filename

        if not path.is_file():

            return False, "document missing before screening"

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return False, "document is not valid UTF-8 text"

        detection = security.detect_prompt_injection(content)

        if detection["detected"]:
            return False, (
                "prompt injection suspected in document content: "
                + ", ".join(detection["matches"])
            )

        if pii_masker.mask_sensitive_text(content) != content:
            return False, (
                "document contains unmasked sensitive data "
                "(email, phone, card, CVV, UPI or account number); "
                "mask it before submitting"
            )

        return True, "security screening passed"

    return PipelinePorts(
        detect=detect,
        activate=activate,
        current_version=current_version,
        rollback=rollback,
        health_check=health_check,
        quality_gate=quality_gate,
        quarantine=quarantine,
        rebuild_index=rebuild_index,
        security_screen=security_screen,
    )


# --------------------------------------------------
# the scheduler
# --------------------------------------------------

class KnowledgeScheduler:
    """
    Drives one activation run at a time.

    Deliberately not a daemon loop. ``activate()`` decides whether this
    instant is a valid moment to run and does the work; the caller (a
    cron entry, a thread, or a test walking simulated time forward)
    decides when to call it. That keeps the schedule a pure function of
    the clock.
    """

    def __init__(
        self,
        ports: PipelinePorts,
        config: Optional[SchedulerConfig] = None,
        monitor: Optional[Callable[[RunRecord], None]] = None,
    ):

        self.ports = ports
        self.config = config or SchedulerConfig()

        # Called once per run, when the run reaches a final status.
        # Optional and silent by default, so importing or testing the
        # scheduler never writes to the real pipeline log.
        self.monitor = monitor

    # ----------------------------------------------
    # monitoring
    # ----------------------------------------------

    def _notify(self, record: RunRecord) -> None:
        """Hand a finished run to the monitor, if one is wired up."""

        if self.monitor is None:

            return

        self.monitor(record)

    # ----------------------------------------------
    # activation
    # ----------------------------------------------

    def activate(
        self,
        now: Optional[datetime] = None,
        force: bool = False,
        attempt_index: int = 0,
        started_at: Optional[datetime] = None,
    ) -> RunRecord:
        """
        One scheduled activation attempt, then report it if final.

        A thin wrapper around ``_activate`` so that the monitor sees
        exactly one event per run. A run that activated is not reported
        here -- its outcome is not confirmed until the health-check
        window has passed, and ``health_check`` reports it then.

        The wrapper also measures the attempt's real wall-clock time
        onto ``record.latency_seconds`` -- the value the monitor's
        latency column is filled from, instead of the literal 0.0 that
        used to be written there.
        """

        started = time.perf_counter()

        record = self._activate(
            now=now,
            force=force,
            attempt_index=attempt_index,
            started_at=started_at,
        )

        record.latency_seconds = time.perf_counter() - started

        if record.status in FINAL_STATUSES:

            self._notify(record)

        return record

    def _activate(
        self,
        now: Optional[datetime] = None,
        force: bool = False,
        attempt_index: int = 0,
        started_at: Optional[datetime] = None,
    ) -> RunRecord:
        """
        One scheduled activation attempt.

        Returns a ``RunRecord`` in every case -- outside the window,
        nothing to do, quarantined, activated, or pending a retry --
        because "did not run" is a result the caller needs to see, not
        an exception.

        ``force`` skips the window check, for an operator running an
        update by hand. It never skips the quality gate.
        """

        config = self.config

        moment = as_utc(now)

        record = RunRecord(
            started_at=started_at or moment,
            status=STATUS_SKIPPED,
        )

        record._health_window_minutes = (
            config.health_check_window_minutes
        )

        if not force and not is_maintenance_window(moment, config):

            record.next_window_at = next_maintenance_start(
                moment, config
            )

            record.notes.append(
                "Outside the maintenance window "
                f"({config.update_time} + "
                f"{config.maintenance_window_minutes} min); "
                "next opening "
                f"{record.next_window_at.isoformat()}."
            )

            # A retry that is not allowed outside the window waits for
            # the next one rather than running now.
            if config.retry_requires_window and attempt_index > 0:

                record.status = STATUS_PENDING_RETRY

                record.next_retry_at = record.next_window_at

            return record

        detected = self.ports.detect() or {}

        record.detected = {
            "new": list(detected.get("new") or []),
            "modified": list(detected.get("modified") or []),
            "unchanged": list(detected.get("unchanged") or []),
        }

        record.skipped_unchanged = record.detected["unchanged"]

        candidates = (
            record.detected["new"] + record.detected["modified"]
        )

        if not candidates:

            record.status = STATUS_NO_CHANGES

            record.finished_at = moment

            record.notes.append(
                "No new or modified documents; nothing to activate."
            )

            return record

        approved: List[str] = []

        for filename in candidates:

            # Security screening comes first: a document carrying
            # prompt-injection instructions or unmasked sensitive data
            # never even reaches the quality gate. Rejected documents
            # are quarantined like any other, with the screen's reason.
            screened, screen_reason = self._screen(filename)

            if not screened:

                record.quarantined.append(filename)

                record.quarantine_reasons[filename] = screen_reason

                if self.ports.quarantine is not None:

                    self.ports.quarantine(filename, screen_reason)

                continue

            if not config.quality_gate_enabled:

                approved.append(filename)

                continue

            passed, message = self._quality(filename)

            if passed:

                approved.append(filename)

                continue

            record.quarantined.append(filename)

            record.quarantine_reasons[filename] = message

            if self.ports.quarantine is not None:

                self.ports.quarantine(filename, message)

        if not approved:

            record.status = STATUS_QUARANTINED

            record.finished_at = moment

            record.notes.append(
                "Every candidate failed the quality gate; the active "
                "version is untouched."
            )

            return record

        failure: Optional[str] = None

        for filename in approved:

            try:

                previous = self.ports.current_version(filename)

                version = self.ports.activate(filename)

            except Exception as error:  # noqa: BLE001

                failure = f"{filename}: {error}"

                break

            record.previous_versions[filename] = previous

            record.versions[filename] = version

            record.activated.append(filename)

        # An activation the serving index never sees is not an
        # activation: the metadata row and the retrieval service must
        # not drift apart. Rebuild the index now that every gated
        # document is in, and treat a rebuild failure exactly like an
        # activation failure -- roll back and retry.
        if (
            failure is None
            and record.activated
            and config.rebuild_index_enabled
        ):

            try:
                self._rebuild_index()

            except Exception as error:  # noqa: BLE001

                failure = f"serving index rebuild failed: {error}"

        if failure is not None:

            # Roll back the ones that did go in, so a partial
            # activation cannot leave the knowledge base half updated.
            for filename in list(record.activated):

                try:

                    self.ports.rollback(
                        filename, record.previous_versions[filename]
                    )

                    record.activated.remove(filename)

                    record.notes.append(
                        f"Reverted partial activation of {filename}."
                    )

                except Exception as error:  # noqa: BLE001

                    record.notes.append(
                        f"Could not revert {filename}: {error}"
                    )

            self._mark_failure(record, moment, attempt_index, failure)

            return record

        record.finished_at = moment

        record.status = STATUS_ACTIVATED

        if record.quarantined:

            record.notes.append(
                "Activated "
                f"{len(record.activated)} document(s); "
                f"{len(record.quarantined)} quarantined."
            )

        return record

    # ----------------------------------------------
    # retries
    # ----------------------------------------------

    def retry(
        self,
        record: RunRecord,
        now: Optional[datetime] = None,
        force: bool = False,
    ) -> RunRecord:
        """
        Run the next retry if it is due.

        A call before the due instant changes nothing and says so,
        which is what a polling caller needs: the retry happens at the
        scheduled minute, not at the first tick after the failure.
        """

        config = self.config

        moment = as_utc(now)

        if record.status not in (STATUS_PENDING_RETRY, STATUS_FAILED):

            record.notes.append(
                f"No retry: run status is {record.status}."
            )

            return record

        due = record.next_retry_at

        if due is None:

            record.notes.append(
                "No retry: the retry schedule is exhausted."
            )

            return record

        if not force and moment < due:

            record.notes.append(
                f"Retry not due until {due.isoformat()}."
            )

            return record

        if (
            config.retry_requires_window
            and not force
            and not is_maintenance_window(moment, config)
        ):

            record.next_retry_at = next_maintenance_start(
                moment, config
            )

            record.notes.append(
                "Retry deferred to the next maintenance window at "
                f"{record.next_retry_at.isoformat()}."
            )

            return record

        attempt = self.activate(
            now=moment,
            # With retry_requires_window off, the retry is explicitly
            # allowed to activate outside the window -- that is what
            # the setting means. Leaving it on would make the setting
            # do nothing, because activate() enforces the window
            # itself and the retry would come back "skipped".
            force=force or not config.retry_requires_window,
            attempt_index=record.attempted,
            started_at=record.started_at,
        )

        # The retry reuses this record rather than returning a new one,
        # so the caller keeps one object holding the whole history.
        attempt.attempts = record.attempts + attempt.attempts

        attempt.health_checks = list(record.health_checks)

        attempt.quarantined = list(
            dict.fromkeys(record.quarantined + attempt.quarantined)
        )

        attempt.quarantine_reasons = {
            **record.quarantine_reasons,
            **attempt.quarantine_reasons,
        }

        attempt._health_window_minutes = (
            config.health_check_window_minutes
        )

        return attempt

    def _mark_failure(
        self,
        record: RunRecord,
        moment: datetime,
        attempt_index: int,
        detail: str,
    ) -> None:
        """Record a failed activation and schedule the next retry."""

        config = self.config

        delays = config.retry_delays()

        following = (
            delays[attempt_index]
            if attempt_index < len(delays)
            else None
        )

        record.attempts.append(
            UpdateAttempt(
                index=attempt_index,
                attempted_at=moment,
                outcome="error",
                detail=detail,
                retry_after_minutes=following,
            )
        )

        record.finished_at = moment

        if following is None:

            record.status = STATUS_FAILED

            record.next_retry_at = None

            record.notes.append(
                "All retries exhausted; the active version is "
                "unchanged."
            )

            return

        record.status = STATUS_PENDING_RETRY

        record.next_retry_at = moment + timedelta(minutes=following)

        record.notes.append(
            f"Activation failed; retrying in {following} minutes at "
            f"{record.next_retry_at.isoformat()}."
        )

    # ----------------------------------------------
    # post-activation health
    # ----------------------------------------------

    def health_check(
        self,
        record: RunRecord,
        now: Optional[datetime] = None,
        force: bool = False,
    ) -> RunRecord:
        """
        Run the health check due at this instant, if any.

        Rolls back the activation automatically when a check fails
        inside the health-check window. Once the window has elapsed
        without a failure the run is closed as healthy and no further
        checks are made, which is what makes the rollback deadline a
        deadline rather than an open-ended watch.
        """

        config = self.config

        moment = as_utc(now)

        record._health_window_minutes = (
            config.health_check_window_minutes
        )

        if record.status != STATUS_ACTIVATED:

            record.notes.append(
                f"No health check: run status is {record.status}."
            )

            return record

        if record.monitoring_complete:

            return record

        activated_at = record.finished_at or record.started_at

        elapsed = (
            moment - activated_at
        ).total_seconds() / 60.0

        # How many probes are due by now, given the interval.
        due_count = int(
            elapsed // config.health_check_interval_minutes
        )

        while (
            len(record.health_checks) < due_count
            and len(record.health_checks)
            < config.health_check_window_minutes
            // config.health_check_interval_minutes
        ):

            checked_at = activated_at + timedelta(
                minutes=config.health_check_interval_minutes
                * (len(record.health_checks) + 1)
            )

            healthy, detail = self._probe(record)

            record.health_checks.append(
                HealthCheck(
                    checked_at=checked_at,
                    minutes_after_activation=round(
                        (
                            checked_at - activated_at
                        ).total_seconds() / 60.0,
                        4,
                    ),
                    healthy=healthy,
                    detail=detail,
                )
            )

            if not healthy:

                # Roll back at `moment`, not at the probe's scheduled
                # instant. The probe may be one the caller is only now
                # catching up on, and the run must not claim to have
                # acted earlier than it did: if monitoring was lazy,
                # the rollback really was late and the deadline check
                # in rolled_back_within_window has to say so.
                self._auto_rollback(record, moment, detail)

                self._notify(record)

                return record

        if elapsed >= config.health_check_window_minutes:

            record.monitoring_complete = True

            record.notes.append(
                "Health check window of "
                f"{config.health_check_window_minutes} minute(s) "
                "elapsed with no failure."
            )

            # The run is only really confirmed once the window has
            # passed, so this is when a healthy activation is reported.
            self._notify(record)

        return record

    def _probe(self, record: RunRecord) -> Tuple[bool, str]:
        """Health-check every activated document."""

        for filename in record.activated:

            try:

                healthy, detail = self.ports.health_check(filename)

            except Exception as error:  # noqa: BLE001

                return False, f"{filename}: {error}"

            if not healthy:

                return False, f"{filename}: {detail}"

        return True, "ok"

    def _auto_rollback(
        self,
        record: RunRecord,
        acted_at: datetime,
        detail: str,
    ) -> None:
        """
        Restore the versions this run replaced.

        ``rollback_minutes`` is measured from activation to the instant
        the rollback was carried out, so the five-minute guarantee is a
        property of the record and can be asserted directly instead of
        inferred from the log.
        """

        activated_at = record.finished_at or record.started_at

        for filename in list(record.activated):

            target = record.previous_versions.get(filename, 0)

            try:

                self.ports.rollback(filename, target)

                record.rolled_back.append(filename)

                record.activated.remove(filename)

            except Exception as error:  # noqa: BLE001

                record.notes.append(
                    f"Rollback of {filename} failed: {error}"
                )

        # The serving index still holds the bad activation; it must
        # serve the restored versions again. Best effort -- the
        # rollback itself has already succeeded -- but a failure is
        # recorded rather than silently leaving a stale index.
        if record.rolled_back:

            try:
                self._rebuild_index()

            except Exception as error:  # noqa: BLE001

                record.notes.append(
                    f"Serving index rebuild after rollback failed: {error}"
                )

        record.rollback_reason = detail

        record.rollback_minutes = round(
            (acted_at - activated_at).total_seconds() / 60.0, 4
        )

        record.monitoring_complete = True

        record.status = STATUS_ROLLED_BACK

        record.notes.append(
            f"Health check failed after "
            f"{record.rollback_minutes} minute(s); rolled back to "
            f"{record.previous_versions}."
        )

    # ----------------------------------------------

    def _quality(self, filename: str) -> Tuple[bool, str]:
        """Run the quality gate, treating a missing one as passthrough."""

        if self.ports.quality_gate is None:

            return True, "no quality gate configured"

        try:

            return self.ports.quality_gate(filename)

        except Exception as error:  # noqa: BLE001

            return False, f"quality gate error: {error}"

    def _screen(self, filename: str) -> Tuple[bool, str]:
        """
        Run the security screen.

        A screen that is not wired up is a passthrough (tests and
        minimal deployments), but a screen that *errors* fails closed:
        an unreadable document is not "probably fine".
        """

        if (
            not self.config.security_screen_enabled
            or self.ports.security_screen is None
        ):

            return True, "no security screening configured"

        try:

            return self.ports.security_screen(filename)

        except Exception as error:  # noqa: BLE001

            return False, f"security screening error: {error}"

    def _rebuild_index(self) -> None:
        """
        Refresh the serving index; a no-op when no hook is wired.

        Called after a successful activation and again after an
        automatic rollback, so the retrieval service always serves
        what the metadata says is active.
        """

        if self.ports.rebuild_index is None:

            return

        self.ports.rebuild_index()


# --------------------------------------------------
# monitoring bridge
# --------------------------------------------------

_MONITOR_MODULE = None


def _monitor_module():
    """
    ``monitor.py``, importable wherever the scheduler was loaded from.

    Plain ``import monitor`` works when this directory is on sys.path
    (running the file, or the scheduler test); when the scheduler was
    itself loaded by path -- how an app worker wires it in -- that
    import fails, so fall back to loading the file by path.
    """

    global _MONITOR_MODULE

    if _MONITOR_MODULE is not None:
        return _MONITOR_MODULE

    try:
        import monitor as module  # noqa: S107 - sibling module

    except ImportError:
        module = _load_module(
            "kp_monitor_by_scheduler",
            Path(__file__).resolve().parent / "monitor.py",
        )

    _MONITOR_MODULE = module

    return module


def _measured_latency(record: RunRecord) -> float:
    """
    A real elapsed time for the run, in seconds.

    The scheduler's own wall-clock measurement when it ran the attempt
    (``record.latency_seconds``), otherwise the run's own timestamps.
    Zero only when nothing was measured at all -- never the old
    unconditional literal.
    """

    if record.latency_seconds is not None:

        return round(record.latency_seconds, 6)

    if record.finished_at is not None:

        elapsed = (
            record.finished_at - record.started_at
        ).total_seconds()

        return max(0.0, elapsed)

    return 0.0


def _measured_confidence(record: RunRecord) -> float:
    """
    A confidence the log can actually stand behind.

    In priority order: the share of the run's health checks that were
    healthy (the strongest evidence, once the window has run); the
    share of candidates that passed the gates (a quarantined-only run
    is a real 0, not a default); 1.0 for a run that found nothing to
    do and did it cleanly; 0.0 for a skipped run, which verified
    nothing.
    """

    if record.health_checks:

        healthy = sum(
            1 for check in record.health_checks if check.healthy
        )

        return healthy / len(record.health_checks)

    candidates = len(record.activated) + len(record.quarantined)

    if candidates:

        return len(record.activated) / candidates

    if record.status == STATUS_NO_CHANGES:

        return 1.0

    return 0.0


def log_run(
    record: RunRecord,
    db_path: Optional[Path] = None,
    support_db: Optional[Path] = None,
    latency: Optional[float] = None,
    confidence: Optional[float] = None,
) -> None:
    """
    Record a finished run in the pipeline monitor.

    ``escalations`` is left out deliberately. The monitor used to be
    called with a hardcoded ``0``, which is why the column never said
    anything; it now reads the real count from the support store, so
    the log answers "how many escalations was this update accompanied
    by" instead of asserting zero.

    ``latency`` and ``confidence`` are likewise derived from the run
    when not given: the attempt's measured wall-clock time and the
    health-check / gate pass ratio, instead of the literal 0.0 that
    used to be written for every run.

    This is the default ``monitor`` of the ``scheduler()`` entry
    point; ``KnowledgeScheduler(ports, config, monitor=log_run)`` for
    hand-built instances.
    """

    files = ", ".join(
        record.activated
        or record.quarantined
        or list(record.detected.get("new") or [])
        or ["-"]
    )

    _monitor_module().log_pipeline_event(
        filename=files,
        status=record.status,
        latency=latency if latency is not None else _measured_latency(record),
        confidence=(
            confidence
            if confidence is not None
            else _measured_confidence(record)
        ),
        escalations=None,
        now=record.finished_at or record.started_at,
        support_db=support_db,
        db_path=db_path,
    )


# --------------------------------------------------
# convenience: a single tick
# --------------------------------------------------

def scheduler(
    now: Optional[datetime] = None,
    ports: Optional[PipelinePorts] = None,
    config: Optional[SchedulerConfig] = None,
    monitor: Optional[Callable[[RunRecord], None]] = log_run,
) -> RunRecord:
    """
    One tick, in the shape the old ``scheduler()`` had.

    Kept as a function because that is the name the previous stub
    exposed; it now returns the run record instead of printing.

    ``monitor`` defaults to ``log_run``: this entry point is the one
    the process actually runs, so a scheduled tick is recorded in the
    pipeline log by default -- the log used to exist only in tests.
    Pass a callable to log elsewhere, or ``None`` to stay silent.
    """

    scheduler_instance = KnowledgeScheduler(
        ports or default_ports(), config, monitor=monitor
    )

    return scheduler_instance.activate(now)


# --------------------------------------------------
# self-test / demo
# --------------------------------------------------

if __name__ == "__main__":

    demo = SchedulerConfig(
        update_time="02:00",
        maintenance_window_minutes=60,
        calendar=BusinessCalendarConfig(timezone_name="UTC"),
    )

    friday_evening = datetime(
        2026, 10, 2, 20, 0, tzinfo=timezone.utc
    )

    print("Update time (local):", demo.update_time)
    print("Maintenance window:", demo.maintenance_window_minutes, "min")
    print("Retry delays:", demo.retry_delays())
    print()

    print("Friday 20:00 UTC in a window?",
          is_maintenance_window(friday_evening, demo))
    print("Next window opens:",
          next_maintenance_start(friday_evening, demo).isoformat())
    print()

    print("Retries after a Friday 02:10 failure:")
    for moment in retry_schedule(
        datetime(2026, 10, 2, 2, 10, tzinfo=timezone.utc), demo
    ):
        print("  ", moment.isoformat())

    print()
    print("Health-check deadline:",
          demo.health_check_window_minutes, "minute(s)")
