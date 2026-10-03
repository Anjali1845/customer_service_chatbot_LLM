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
3. Puts every candidate through the quality gate *before* activation.
   A document that fails is quarantined with its reason instead of
   being activated and rolled back later.
4. Activates, remembering the version it replaced.
5. Watches the new version with a health check for
   ``health_check_window_minutes`` (5 by default) and **auto-rolls back
   to the previous version if a check fails**.
6. Retries a failed activation at 15, then 30, then 60 minutes, and
   gives up after the last one instead of looping forever.

Everything is time-injected (``now=``) and dependency-injected
(``PipelinePorts``), so the whole schedule -- windows, retries, the
rollback deadline -- is exercisable without sleeping, without a real
database and without touching the live knowledge base. The test suite
drives simulated time through these same entry points.

All public functions take and return timezone-aware UTC datetimes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


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
    """

    detect: Callable[[], Dict[str, List[str]]]
    activate: Callable[[str], int]
    current_version: Callable[[str], int]
    rollback: Callable[[str, int], None]
    health_check: Callable[[str], Tuple[bool, str]]
    quality_gate: Optional[Callable[[str], Tuple[bool, str]]] = None
    quarantine: Optional[Callable[[str, str], None]] = None


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


def default_ports(
    documents_dir: Optional[Path] = None,
    metadata_db: Optional[Path] = None,
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
    """

    import sqlite3

    pipeline_dir = Path(__file__).resolve().parent

    documents_dir = (
        documents_dir
        or (PROJECT_ROOT / "knowledge_base" / "documents")
    )
    metadata_db = metadata_db or (PROJECT_ROOT / "metadata.db")

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

        # The real probe: the document must still be readable and must
        # still pass the quality gate. Anything that made it past the
        # gate and cannot now be read is a bad activation.
        path = Path(documents_dir) / filename

        if not path.is_file():

            return False, "document missing after activation"

        passed, message = quality_validator.validate_document(path)

        return passed, message

    def quality_gate(filename: str) -> Tuple[bool, str]:

        # The ports contract passes a filename; validate_document takes
        # a path. Resolving it here is what keeps the gate pointed at
        # the documents directory rather than at the working directory,
        # which is where a bare filename would otherwise land.
        return quality_validator.validate_document(
            Path(documents_dir) / filename
        )

    def quarantine(filename: str, reason: str) -> None:

        print(f"🚧 Quarantined {filename}: {reason}")

    return PipelinePorts(
        detect=detect,
        activate=activate,
        current_version=current_version,
        rollback=rollback,
        health_check=health_check,
        quality_gate=quality_gate,
        quarantine=quarantine,
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
        """

        record = self._activate(
            now=now,
            force=force,
            attempt_index=attempt_index,
            started_at=started_at,
        )

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


# --------------------------------------------------
# monitoring bridge
# --------------------------------------------------

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

    Wire this in with ``KnowledgeScheduler(ports, config,
    monitor=log_run)``.
    """

    from monitor import log_pipeline_event

    files = ", ".join(
        record.activated
        or record.quarantined
        or list(record.detected.get("new") or [])
        or ["-"]
    )

    log_pipeline_event(
        filename=files,
        status=record.status,
        latency=latency if latency is not None else 0.0,
        confidence=confidence if confidence is not None else 0.0,
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
) -> RunRecord:
    """
    One tick, in the shape the old ``scheduler()`` had.

    Kept as a function because that is the name the previous stub
    exposed; it now returns the run record instead of printing.
    """

    scheduler_instance = KnowledgeScheduler(
        ports or default_ports(), config
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
