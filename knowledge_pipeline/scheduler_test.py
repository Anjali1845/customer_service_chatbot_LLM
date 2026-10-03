"""
KNOWLEDGE BASE SCHEDULER TEST SUITE

Covers:
- Config validation (a typo must not silently disable updates)
- Maintenance window boundaries, weekends, holidays, wrap past midnight
- Runtime changes to the window
- Retry schedule at exactly 15 / 30 / 60 minutes, then exhaustion
- Retry deferral when retries are constrained to the window
- Quality gate before activation, quarantine, active version untouched
- Partial activation reverted when a later document fails
- Health check passing, and failing inside the 5-minute deadline
  triggering an automatic rollback
- force=True bypassing the window but never the quality gate

Runs entirely on simulated time against a fake pipeline: no sleeping,
no metadata.db, no documents on disk.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sqlite3
import tempfile

from monitor import escalation_count, escalation_summary

from scheduler import (
    STATUS_ACTIVATED,
    STATUS_FAILED,
    STATUS_NO_CHANGES,
    STATUS_PENDING_RETRY,
    STATUS_QUARANTINED,
    STATUS_ROLLED_BACK,
    STATUS_SKIPPED,
    BusinessCalendarConfig,
    KnowledgeScheduler,
    PipelinePorts,
    SchedulerConfig,
    is_maintenance_window,
    log_run,
    default_ports,
    next_maintenance_start,
    next_retry_at,
    retry_schedule,
    window_closes_at,
)


PASS = 0
FAIL = 0


def check(name, condition, details=""):
    global PASS, FAIL

    if condition:
        PASS += 1
        print(f"✅ {name}")
    else:
        FAIL += 1
        print(f"❌ {name}")
        if details:
            print(f"   {details}")


def utc(year, month, day, hour=0, minute=0):
    return datetime(
        year, month, day, hour, minute, tzinfo=timezone.utc
    )


def config(**overrides):
    """UTC calendar, 02:00 window, so the arithmetic is readable."""

    overrides.setdefault(
        "calendar", BusinessCalendarConfig(timezone_name="UTC")
    )

    return SchedulerConfig(**overrides)


# --------------------------------------------------
# fake pipeline
# --------------------------------------------------

class FakePipeline:
    """
    An in-memory stand-in for detect/validate/activate/rollback.

    Records every call so the test can assert what the scheduler did
    as well as what it reported.
    """

    def __init__(
        self,
        new=(),
        modified=(),
        unchanged=(),
        gate_fail=(),
        activate_error=(),
        versions=None,
        activate_next=(),
    ):
        self.new = list(new)
        self.modified = list(modified)
        self.unchanged = list(unchanged)
        self.gate_fail = set(gate_fail)
        self.activate_error = set(activate_error)

        # filename -> already-active version
        self.versions = dict(versions or {})

        # Optional queue of (filename, exception) applied in order.
        self.activate_next = list(activate_next)

        self.healthy = True
        self.health_detail = "ok"

        self.detect_calls = 0
        self.activations = []
        self.rollbacks = []
        self.quarantine_calls = []
        self.gate_calls = []
        self.probes = 0

    # ----------------------------------------------

    def detect(self):
        self.detect_calls += 1

        return {
            "new": list(self.new),
            "modified": list(self.modified),
            "unchanged": list(self.unchanged),
        }

    def quality_gate(self, filename):
        self.gate_calls.append(filename)

        if filename in self.gate_fail:
            return False, "Document too short"

        return True, "Quality validation passed"

    def current_version(self, filename):
        return self.versions.get(filename, 0)

    def activate(self, filename):
        if self.activate_next:
            target, error = self.activate_next[0]

            if target == filename:
                self.activate_next.pop(0)

                raise error

        if filename in self.activate_error:
            raise RuntimeError("database is locked")

        version = self.versions.get(filename, 0) + 1

        self.versions[filename] = version

        self.activations.append((filename, version))

        return version

    def rollback(self, filename, target_version):
        self.rollbacks.append((filename, target_version))

        self.versions[filename] = target_version

    def health_check(self, filename):
        self.probes += 1

        return self.healthy, self.health_detail

    def quarantine(self, filename, reason):
        self.quarantine_calls.append((filename, reason))

    # ----------------------------------------------

    def ports(self):
        return PipelinePorts(
            detect=self.detect,
            activate=self.activate,
            current_version=self.current_version,
            rollback=self.rollback,
            health_check=self.health_check,
            quality_gate=self.quality_gate,
            quarantine=self.quarantine,
        )

    def build(self, cfg=None):
        return KnowledgeScheduler(self.ports(), cfg or config())


# --------------------------------------------------
# reference times (UTC)
# --------------------------------------------------

MON = utc(2026, 10, 5)          # Monday
MON_0200 = utc(2026, 10, 5, 2, 0)
MON_0201 = utc(2026, 10, 5, 2, 1)
MON_0210 = utc(2026, 10, 5, 2, 10)
MON_0250 = utc(2026, 10, 5, 2, 50)
MON_0259 = utc(2026, 10, 5, 2, 59)
MON_0300 = utc(2026, 10, 5, 3, 0)
MON_1200 = utc(2026, 10, 5, 12, 0)

SAT_0200 = utc(2026, 10, 3, 2, 0)
SUN_0200 = utc(2026, 10, 4, 2, 0)


print("=" * 75)
print("KNOWLEDGE BASE SCHEDULER TESTS")
print("=" * 75)


# ============================================================
# [1] CONFIGURATION VALIDATION
# ============================================================

print("\n[1] CONFIGURATION VALIDATION")

try:
    config(update_time="6pm")
    check("a malformed update time is rejected", False, "no error raised")
except ValueError:
    check("a malformed update time is rejected", True)

try:
    config(maintenance_window_minutes=0)
    check("a zero-length window is rejected", False, "no error raised")
except ValueError:
    check("a zero-length window is rejected", True)

try:
    config(retry_minutes=())
    check("an empty retry schedule is rejected", False, "no error raised")
except ValueError:
    check("an empty retry schedule is rejected", True)

try:
    config(retry_minutes=(30, 15, 60))
    check(
        "out-of-order retry delays are rejected", False, "no error raised"
    )
except ValueError:
    check("out-of-order retry delays are rejected", True)

try:
    config(health_check_window_minutes=0)
    check(
        "a zero health-check window is rejected", False, "no error raised"
    )
except ValueError:
    check("a zero health-check window is rejected", True)

cfg = config()
check(
    "the default retry delays are 15/30/60 minutes",
    cfg.retry_delays() == (15, 30, 60),
    str(cfg.retry_delays()),
)
check(
    "the default health-check deadline is 5 minutes",
    cfg.health_check_window_minutes == 5,
)
check(
    "retry_delays normalises a list assigned at runtime to a tuple",
    config(retry_minutes=[15, 30, 60]).retry_delays() == (15, 30, 60),
    str(config(retry_minutes=[15, 30, 60]).retry_delays()),
)
check(
    "max_attempts counts the original run plus the retries",
    cfg.max_attempts() == 4,
    str(cfg.max_attempts()),
)


# ============================================================
# [2] MAINTENANCE WINDOW
# ============================================================

print("\n[2] MAINTENANCE WINDOW")

cfg = config(update_time="02:00", maintenance_window_minutes=60)
check("the window opening instant is inside", is_maintenance_window(MON_0200, cfg))
check("a minute into the window is inside", is_maintenance_window(MON_0201, cfg))
check(
    "the last minute before it closes is inside",
    is_maintenance_window(MON_0259, cfg),
)
check(
    "the closing instant is outside (half-open)",
    not is_maintenance_window(MON_0300, cfg),
)
check("midday is outside", not is_maintenance_window(MON_1200, cfg))
check("one minute before opening is outside", not is_maintenance_window(MON_0200 - timedelta(minutes=1), cfg))

check("Saturday is not a window", not is_maintenance_window(SAT_0200, cfg))
check("Sunday is not a window", not is_maintenance_window(SUN_0200, cfg))

check(
    "the next window from Friday evening is Monday",
    next_maintenance_start(utc(2026, 10, 2, 20, 0), cfg) == MON_0200,
    str(next_maintenance_start(utc(2026, 10, 2, 20, 0), cfg)),
)

check(
    "the next window is strictly after now, even inside a window",
    next_maintenance_start(MON_0201, cfg) == MON_0200 + timedelta(days=1),
    str(next_maintenance_start(MON_0201, cfg)),
)

check(
    "a closed window reports its closing instant",
    window_closes_at(MON_0201, cfg)
    == MON_0201 + timedelta(minutes=59),
    str(window_closes_at(MON_0201, cfg)),
)
check(
    "a shut window has no closing instant",
    window_closes_at(MON_1200, cfg) is None,
)

# A window that wraps past midnight.
wrapping = config(update_time="23:30", maintenance_window_minutes=60)
check(
    "a window wrapping midnight covers 23:30",
    is_maintenance_window(utc(2026, 10, 5, 23, 30), wrapping),
)
check(
    "a window wrapping midnight covers 00:15",
    is_maintenance_window(utc(2026, 10, 6, 0, 15), wrapping),
)
check(
    "a window wrapping midnight is shut at 00:30",
    not is_maintenance_window(utc(2026, 10, 6, 0, 30), wrapping),
)

# A configured holiday.
holiday_cfg = config(
    calendar=BusinessCalendarConfig(
        timezone_name="UTC", holidays=("2026-10-05",)
    )
)
check(
    "a configured holiday is not a window",
    not is_maintenance_window(MON_0200, holiday_cfg),
)
check(
    "a holiday rolls the window to the next working day",
    next_maintenance_start(MON_0200, holiday_cfg)
    == utc(2026, 10, 6, 2, 0),
    str(next_maintenance_start(MON_0200, holiday_cfg)),
)

# Working seven days a week.
always = config(
    calendar=BusinessCalendarConfig(
        timezone_name="UTC", exclude_weekends=False
    )
)
check(
    "exclude_weekends=False opens the window on a Saturday",
    is_maintenance_window(SAT_0200, always),
)
check(
    "exclude_weekends=False opens the window on a Sunday",
    is_maintenance_window(SUN_0200, always),
)

# Runtime change to the window.
live = config(update_time="02:00", maintenance_window_minutes=60)
check("before the change, 04:00 is outside", not is_maintenance_window(utc(2026, 10, 5, 4, 0), live))

live.update_time = "04:00"
live.maintenance_window_minutes = 30

check(
    "after moving the window, 04:00 is inside",
    is_maintenance_window(utc(2026, 10, 5, 4, 0), live),
)
check(
    "after widening the window, 02:00 is outside",
    not is_maintenance_window(MON_0200, live),
)


# ============================================================
# [3] RETRY SCHEDULE
# ============================================================

print("\n[3] RETRY SCHEDULE")

cfg = config()

schedule = retry_schedule(MON_0210, cfg)

check(
    "a failure schedules exactly three retries",
    len(schedule) == 3,
    str(len(schedule)),
)
check(
    "the first retry is 15 minutes later",
    schedule[0] == utc(2026, 10, 5, 2, 25),
    str(schedule[0]),
)
check(
    "the second retry is 30 minutes later",
    schedule[1] == utc(2026, 10, 5, 2, 40),
    str(schedule[1]),
)
check(
    "the third retry is 60 minutes later",
    schedule[2] == utc(2026, 10, 5, 3, 10),
    str(schedule[2]),
)
check(
    "the retries are strictly increasing",
    schedule == sorted(schedule),
)

check(
    "attempt 0 schedules the 15-minute retry",
    next_retry_at(MON_0210, 0, cfg) == utc(2026, 10, 5, 2, 25),
)
check(
    "attempt 1 schedules the 30-minute retry",
    next_retry_at(MON_0210, 1, cfg) == utc(2026, 10, 5, 2, 40),
)
check(
    "attempt 2 schedules the 60-minute retry",
    next_retry_at(MON_0210, 2, cfg) == utc(2026, 10, 5, 3, 10),
)
check(
    "attempt 3 has no retry left",
    next_retry_at(MON_0210, 3, cfg) is None,
)

# Retry delays are wall-clock, not business minutes: a Friday-night
# failure retries the same evening rather than waiting for Monday.
friday_failure = utc(2026, 10, 2, 2, 10)
friday_retries = retry_schedule(friday_failure, cfg)
check(
    "retries are wall-clock and do not skip the weekend",
    friday_retries[0] == utc(2026, 10, 2, 2, 25),
    str(friday_retries[0]),
)

custom = config(retry_minutes=(5, 10))
check(
    "a reconfigured retry schedule is honoured",
    retry_schedule(MON_0210, custom)
    == [utc(2026, 10, 5, 2, 15), utc(2026, 10, 5, 2, 20)],
    str(retry_schedule(MON_0210, custom)),
)


# ============================================================
# [4] OUTSIDE THE WINDOW
# ============================================================

print("\n[4] OUTSIDE THE WINDOW")

fake = FakePipeline(new=["faq.txt"])
record = fake.build().activate(MON_1200)

check("a run outside the window is skipped", record.status == STATUS_SKIPPED, record.status)
check("a skipped run activates nothing", fake.activations == [], str(fake.activations))
check("a skipped run does not even detect changes", fake.detect_calls == 0)
check(
    "a skipped run reports when the next window opens",
    record.next_window_at == utc(2026, 10, 6, 2, 0),
    str(record.next_window_at),
)
check(
    "a skipped run explains itself",
    any("maintenance window" in note for note in record.notes),
    str(record.notes),
)

fake = FakePipeline(new=["faq.txt"])
record = fake.build().activate(SAT_0200)

check("a Saturday run is skipped", record.status == STATUS_SKIPPED, record.status)
check(
    "a Saturday run waits for Monday",
    record.next_window_at == MON_0200,
    str(record.next_window_at),
)

fake = FakePipeline(new=["faq.txt"])
record = fake.build().activate(MON_1200, force=True)

check("force=True runs outside the window", record.status == STATUS_ACTIVATED, record.status)
check("force=True activates the document", fake.activations == [("faq.txt", 1)], str(fake.activations))

forced_gate = FakePipeline(new=["bad.txt"], gate_fail=["bad.txt"])
record = forced_gate.build().activate(MON_1200, force=True)

check(
    "force=True still runs the quality gate",
    record.status == STATUS_QUARANTINED,
    record.status,
)
check(
    "force=True cannot activate a document that fails the gate",
    forced_gate.activations == [],
    str(forced_gate.activations),
)


# ============================================================
# [5] QUALITY GATE AND QUARANTINE
# ============================================================

print("\n[5] QUALITY GATE AND QUARANTINE")

fake = FakePipeline(
    new=["good.txt", "bad.txt"], gate_fail=["bad.txt"]
)
record = fake.build().activate(MON_0200)

check("a gated run activates", record.status == STATUS_ACTIVATED, record.status)
check("the passing document is activated", record.activated == ["good.txt"], str(record.activated))
check("the failing document is quarantined", record.quarantined == ["bad.txt"], str(record.quarantined))
check(
    "the quarantine reason is recorded",
    record.quarantine_reasons.get("bad.txt") == "Document too short",
    str(record.quarantine_reasons),
)
check(
    "the failing document is never activated",
    fake.activations == [("good.txt", 1)],
    str(fake.activations),
)
check(
    "the quarantine hook is called with the reason",
    fake.quarantine_calls == [("bad.txt", "Document too short")],
    str(fake.quarantine_calls),
)
check(
    "the gate is run before activation",
    fake.gate_calls == ["good.txt", "bad.txt"],
    str(fake.gate_calls),
)

all_bad = FakePipeline(
    new=["bad.txt"], modified=["worse.txt"],
    gate_fail=["bad.txt", "worse.txt"],
    versions={"worse.txt": 3},
)
record = all_bad.build().activate(MON_0200)

check(
    "nothing passing the gate leaves the run quarantined",
    record.status == STATUS_QUARANTINED,
    record.status,
)
check("nothing is activated", all_bad.activations == [], str(all_bad.activations))
check(
    "the active version is untouched",
    all_bad.versions["worse.txt"] == 3,
    str(all_bad.versions),
)
check(
    "the run says the active version is untouched",
    any("untouched" in note for note in record.notes),
    str(record.notes),
)

# The gate is not run against unchanged documents.
fake = FakePipeline(new=["a.txt"], unchanged=["b.txt", "c.txt"])
record = fake.build().activate(MON_0200)

check(
    "the gate is not run against unchanged documents",
    fake.gate_calls == ["a.txt"],
    str(fake.gate_calls),
)
check(
    "unchanged documents are reported",
    record.skipped_unchanged == ["b.txt", "c.txt"],
    str(record.skipped_unchanged),
)

# Disabling the gate.
open_gate = FakePipeline(new=["bad.txt"], gate_fail=["bad.txt"])
record = open_gate.build(
    config(quality_gate_enabled=False)
).activate(MON_0200)

check(
    "quality_gate_enabled=False activates without gating",
    record.status == STATUS_ACTIVATED,
    record.status,
)
check(
    "a disabled gate runs no quality checks",
    open_gate.gate_calls == [],
    str(open_gate.gate_calls),
)

# No changes at all.
idle = FakePipeline(unchanged=["a.txt"])
record = idle.build().activate(MON_0200)

check("an unchanged knowledge base reports no_changes", record.status == STATUS_NO_CHANGES, record.status)
check("no_changes activates nothing", idle.activations == [])


# ============================================================
# [6] ACTIVATION AND VERSIONS
# ============================================================

print("\n[6] ACTIVATION AND VERSIONS")

fake = FakePipeline(new=["faq.txt"], versions={"faq.txt": 4})
record = fake.build().activate(MON_0200)

check("the new version is recorded", record.versions == {"faq.txt": 5}, str(record.versions))
check(
    "the replaced version is remembered for rollback",
    record.previous_versions == {"faq.txt": 4},
    str(record.previous_versions),
)
check("a healthy activation finishes at the run instant", record.finished_at == MON_0200)
check(
    "a clean activation carries no rollback",
    record.rolled_back == [] and record.rollback_minutes is None,
)

fake = FakePipeline(new=["a.txt", "b.txt"], modified=["c.txt"])
record = fake.build().activate(MON_0200)

check(
    "new and modified documents are both activated",
    sorted(record.activated) == ["a.txt", "b.txt", "c.txt"],
    str(record.activated),
)
check(
    "the detected categories are reported separately",
    record.detected["new"] == ["a.txt", "b.txt"]
    and record.detected["modified"] == ["c.txt"],
    str(record.detected),
)

# A failure part-way through reverts what already went in.
partial = FakePipeline(
    new=["a.txt", "b.txt"],
    activate_error=["b.txt"],
    versions={"a.txt": 1, "b.txt": 7},
)
record = partial.build().activate(MON_0200)

check(
    "a partial activation is not left half-applied",
    record.activated == [],
    str(record.activated),
)
check(
    "the earlier document is rolled back",
    partial.rollbacks == [("a.txt", 1)],
    str(partial.rollbacks),
)
check(
    "the reverted document keeps its old version",
    partial.versions["a.txt"] == 1,
    str(partial.versions),
)
check(
    "the failed document keeps its old version",
    partial.versions["b.txt"] == 7,
    str(partial.versions),
)
check(
    "a partial failure is scheduled for retry",
    record.status == STATUS_PENDING_RETRY,
    record.status,
)
check(
    "the failure records the error",
    "database is locked" in record.attempts[0].detail,
    str(record.attempts),
)


# ============================================================
# [7] RETRY EXECUTION ON SIMULATED TIME
# ============================================================

print("\n[7] RETRY EXECUTION")

fake = FakePipeline(new=["faq.txt"], activate_error=["faq.txt"])
run = fake.build()

record = run.activate(MON_0210)

check("the first failure is attempt 0", record.attempts[0].index == 0, str(record.attempts))
check(
    "the first failure schedules a 15-minute retry",
    record.attempts[0].retry_after_minutes == 15,
    str(record.attempts),
)
check(
    "the next retry is 15 minutes after the failure",
    record.next_retry_at == utc(2026, 10, 5, 2, 25),
    str(record.next_retry_at),
)
check("the run is pending retry", record.status == STATUS_PENDING_RETRY, record.status)

# A tick before the due instant must not retry.
early = run.retry(record, utc(2026, 10, 5, 2, 24))

check("a retry before its due time does not run", early.attempted == 1, str(early.attempted))
check(
    "an early tick says the retry is not due",
    any("not due" in note for note in early.notes),
    str(early.notes),
)

second = run.retry(record, utc(2026, 10, 5, 2, 25))

check("the retry runs at exactly 15 minutes", second.attempted == 2, str(second.attempted))
check(
    "the retry schedules the 30-minute delay",
    second.next_retry_at == utc(2026, 10, 5, 2, 55),
    str(second.next_retry_at),
)

third = run.retry(second, utc(2026, 10, 5, 2, 55))

check("the second retry runs at 30 minutes", third.attempted == 3, str(third.attempted))
check(
    "the second retry schedules the 60-minute delay",
    third.next_retry_at == utc(2026, 10, 5, 3, 55),
    str(third.next_retry_at),
)

fourth = run.retry(third, utc(2026, 10, 5, 3, 55))

check("the third retry runs at 60 minutes", fourth.attempted == 4, str(fourth.attempted))
check("the run gives up after the last retry", fourth.status == STATUS_FAILED, fourth.status)
check("an exhausted run has no next retry", fourth.next_retry_at is None)
check("an exhausted run reports exhaustion", fourth.exhausted)
check(
    "the attempts are numbered 0 to 3",
    [attempt.index for attempt in fourth.attempts] == [0, 1, 2, 3],
    str([attempt.index for attempt in fourth.attempts]),
)

final = run.retry(fourth, utc(2026, 10, 5, 6, 0))

check(
    "a fourth retry never happens",
    final.attempted == 4,
    str(final.attempted),
)
check(
    "the exhausted run says so",
    any("exhausted" in note for note in final.notes),
    str(final.notes),
)
check(
    "an exhausted failure never activated anything",
    fake.activations == [],
    str(fake.activations),
)

# A retry that succeeds.
recovering = FakePipeline(new=["faq.txt"], activate_next=[("faq.txt", RuntimeError("transient"))])
run = recovering.build()

record = run.activate(MON_0210)
check("the transient failure is attempt 0", record.attempts[0].index == 0, str(record.attempts))

recovered = run.retry(record, utc(2026, 10, 5, 2, 25))

check("the retry can succeed", recovered.status == STATUS_ACTIVATED, recovered.status)
check("the successful retry activated the document", recovering.activations == [("faq.txt", 1)], str(recovering.activations))
check(
    "the successful retry keeps the failure history",
    [attempt.index for attempt in recovered.attempts] == [0],
    str(recovered.attempts),
)

# retry_requires_window.
constrained = FakePipeline(new=["faq.txt"], activate_error=["faq.txt"])

# A deliberately narrow window, so the 15-minute retry comes due after
# it has closed.
run = constrained.build(
    config(
        maintenance_window_minutes=60,
        retry_requires_window=True,
    )
)

record = run.activate(MON_0250)

check("a failure inside the window is pending retry", record.status == STATUS_PENDING_RETRY, record.status)
check(
    "that failure is due at 03:05, after the window shuts",
    record.next_retry_at == utc(2026, 10, 5, 3, 5),
    str(record.next_retry_at),
)

deferred = run.retry(record, utc(2026, 10, 5, 3, 5))

check(
    "a retry outside the window is deferred when required",
    deferred.attempted == 1,
    str(deferred.attempted),
)
check(
    "the deferred retry waits for the next window",
    deferred.next_retry_at == utc(2026, 10, 6, 2, 0),
    str(deferred.next_retry_at),
)
check(
    "the deferral is explained",
    any("deferred" in note for note in deferred.notes),
    str(deferred.notes),
)

allowed = run.retry(deferred, utc(2026, 10, 6, 2, 0))

check(
    "the deferred retry runs in the next window",
    allowed.attempted == 2,
    str(allowed.attempted),
)

# With the constraint off, a retry does activate outside the window
# rather than quietly coming back skipped.
unconstrained = FakePipeline(new=["faq.txt"], activate_error=["faq.txt"])
run = unconstrained.build(
    config(
        maintenance_window_minutes=60,
        retry_requires_window=False,
    )
)

record = run.activate(MON_0250)
outside = run.retry(record, utc(2026, 10, 5, 3, 5))

check(
    "a retry outside the window runs when the constraint is off",
    outside.attempted == 2,
    str(outside.attempted),
)
check(
    "an unconstrained retry is not reported as skipped",
    outside.status != STATUS_SKIPPED,
    outside.status,
)


# ============================================================
# [8] POST-ACTIVATION HEALTH CHECK
# ============================================================

print("\n[8] POST-ACTIVATION HEALTH CHECK")

fake = FakePipeline(new=["faq.txt"])
run = fake.build()

record = run.activate(MON_0200)

check("the activation is healthy at first", record.status == STATUS_ACTIVATED, record.status)

record = run.health_check(record, MON_0201)

check("a health check runs one minute after activation", len(record.health_checks) == 1, str(len(record.health_checks)))
check("the check is recorded as healthy", record.health_checks[0].healthy)
check(
    "the check reports its offset from activation",
    record.health_checks[0].minutes_after_activation == 1.0,
    str(record.health_checks[0].minutes_after_activation),
)
check("a passing check does not roll back", record.rolled_back == [], str(record.rolled_back))
check("the run is still activated", record.status == STATUS_ACTIVATED, record.status)
check("the probe actually ran", fake.probes > 0)

record = run.health_check(record, MON_0200 + timedelta(minutes=5))

check(
    "the window elapsing closes monitoring",
    record.monitoring_complete,
)
check(
    "a healthy run reports five checks",
    len(record.health_checks) == 5,
    str(len(record.health_checks)),
)
check(
    "a healthy run is never rolled back",
    record.rollback_minutes is None and record.rolled_back == [],
)
check(
    "no checks are made after the window closes",
    len(run.health_check(record, MON_0200 + timedelta(minutes=30)).health_checks) == 5,
)

# A check that fails inside the window.
failing = FakePipeline(new=["faq.txt"], versions={"faq.txt": 2})
run = failing.build()

record = run.activate(MON_0200)

check("the failing run starts activated", record.status == STATUS_ACTIVATED, record.status)

record = run.health_check(record, MON_0201)

check("a first healthy check keeps the run", record.status == STATUS_ACTIVATED, record.status)

failing.healthy = False
failing.health_detail = "document missing after activation"

record = run.health_check(record, utc(2026, 10, 5, 2, 3))

check("a failing health check rolls back automatically", record.status == STATUS_ROLLED_BACK, record.status)
check(
    "the rollback meets the 5-minute deadline",
    record.rolled_back_within_window,
    str(record.rollback_minutes),
)
check(
    "the rollback is timed from when it actually acted",
    record.rollback_minutes == 3.0,
    str(record.rollback_minutes),
)
check("the document is restored to its previous version", failing.versions["faq.txt"] == 2, str(failing.versions))
check("the rollback is recorded", record.rolled_back == ["faq.txt"], str(record.rolled_back))
check(
    "the rollback reason is the health-check detail",
    "missing after activation" in record.rollback_reason,
    record.rollback_reason,
)
check("the failed activation is no longer active", record.activated == [], str(record.activated))
check("the run is closed after rollback", record.monitoring_complete)
check(
    "no further checks run after a rollback",
    len(run.health_check(record, utc(2026, 10, 5, 2, 4)).health_checks) == len(record.health_checks),
)
check(
    "a rolled-back run is not reported as activated",
    record.status != STATUS_ACTIVATED,
)

# The deadline is measured from activation, not from the first check.
late = FakePipeline(new=["faq.txt"])
run = late.build()

record = run.activate(MON_0200)

late.healthy = False
record = run.health_check(record, utc(2026, 10, 5, 2, 5))

check(
    "a failure at the edge of the window still rolls back",
    record.status == STATUS_ROLLED_BACK,
    record.status,
)
check(
    "a failure at the edge of the window is inside the deadline",
    record.rollback_minutes <= 5,
    str(record.rollback_minutes),
)

# A rollback that cannot be measured is not claimed to be on time.
check(
    "an activated run is never reported as rolled back in time",
    KnowledgeScheduler(
        FakePipeline(new=["a.txt"]).ports(), config()
    ).activate(MON_0200).rolled_back_within_window
    is False,
)

# A health check on a run that never activated is a no-op.
skipped = FakePipeline(new=["faq.txt"])
run = skipped.build()

record = run.activate(MON_1200)
checked = run.health_check(record, MON_1200)

check("a skipped run is never health-checked", checked.health_checks == [], str(checked.health_checks))
check(
    "a skipped run says why there is no check",
    any("No health check" in note for note in checked.notes),
    str(checked.notes),
)

# The live config decides the deadline.
narrow = FakePipeline(new=["faq.txt"])
run = narrow.build(config(health_check_window_minutes=2))

record = run.activate(MON_0200)

narrow.healthy = False
record = run.health_check(record, utc(2026, 10, 5, 2, 2))

check(
    "a narrowed health-check window is honoured",
    record.status == STATUS_ROLLED_BACK,
    record.status,
)
check(
    "a narrowed window still meets its own deadline",
    record.rolled_back_within_window,
    str(record.rollback_minutes),
)

# A caller that polls late really has rolled back late, and the record
# must not be able to claim otherwise by measuring from the probe's
# scheduled instant instead of from the rollback itself.
lazy = FakePipeline(new=["faq.txt"])
run = lazy.build()

record = run.activate(MON_0200)

lazy.healthy = False
record = run.health_check(record, utc(2026, 10, 5, 2, 30))

check(
    "a late health check still rolls back",
    record.status == STATUS_ROLLED_BACK,
    record.status,
)
check(
    "a late rollback does not claim to have met the deadline",
    not record.rolled_back_within_window,
    str(record.rollback_minutes),
)
check(
    "a late rollback reports its real elapsed time",
    record.rollback_minutes == 30.0,
    str(record.rollback_minutes),
)


# ============================================================
# [9] MONITORING BRIDGE
# ============================================================

print("\n[9] MONITORING BRIDGE")

# The monitor used to be handed a literal 0 escalations by every
# caller. The scheduler must report each run exactly once, at the
# point its outcome is actually known.
reported = []


def spy(record):
    reported.append(record.status)


fake = FakePipeline(new=["faq.txt"])
run = KnowledgeScheduler(fake.ports(), config(), monitor=spy)

record = run.activate(MON_0200)

check(
    "an activation is not reported until its health window closes",
    reported == [],
    str(reported),
)

record = run.health_check(record, MON_0200 + timedelta(minutes=5))

check("a confirmed activation is reported once", reported == [STATUS_ACTIVATED], str(reported))
check(
    "further health checks do not report again",
    run.health_check(record, MON_0200 + timedelta(minutes=9)) is record
    and reported == [STATUS_ACTIVATED],
    str(reported),
)

# A quarantined run reports immediately: its outcome is settled.
reported.clear()

blocked = FakePipeline(new=["bad.txt"], gate_fail=["bad.txt"])
run = KnowledgeScheduler(blocked.ports(), config(), monitor=spy)

run.activate(MON_0200)

check(
    "a quarantined run is reported at once",
    reported == [STATUS_QUARANTINED],
    str(reported),
)

# A rollback replaces the activation, it does not add to it.
reported.clear()

broken = FakePipeline(new=["faq.txt"])
run = KnowledgeScheduler(broken.ports(), config(), monitor=spy)

record = run.activate(MON_0200)

broken.healthy = False

record = run.health_check(record, utc(2026, 10, 5, 2, 2))

check(
    "a rollback is the single event reported for that run",
    reported == [STATUS_ROLLED_BACK],
    str(reported),
)

# An exhausted retry ladder reports once, at the end.
reported.clear()

exhausting = FakePipeline(new=["faq.txt"], activate_error=["faq.txt"])
run = KnowledgeScheduler(exhausting.ports(), config(), monitor=spy)

record = run.activate(MON_0210)

check(
    "a run pending retry is not yet reported",
    reported == [],
    str(reported),
)

record = run.retry(record, utc(2026, 10, 5, 2, 25))
record = run.retry(record, utc(2026, 10, 5, 2, 55))
record = run.retry(record, utc(2026, 10, 5, 3, 55))

check(
    "an exhausted retry ladder reports once, at the end",
    reported == [STATUS_FAILED],
    str(reported),
)

# No monitor wired up means no side effects at all.
quiet = FakePipeline(new=["faq.txt"])
KnowledgeScheduler(quiet.ports(), config()).activate(MON_0200)

check("a scheduler with no monitor writes nothing", quiet.activations == [("faq.txt", 1)], str(quiet.activations))


# ============================================================
# [10] REAL ESCALATION COUNTS IN THE LOG
# ============================================================

print("\n[10] REAL ESCALATION COUNTS IN THE LOG")

# The whole point of the monitor fix: the log line must carry the
# escalation count that actually happened, not the literal 0 that
# every caller used to pass.
with tempfile.TemporaryDirectory() as temp_dir:

    temp = Path(temp_dir)

    support_db = temp / "support.db"

    connection = sqlite3.connect(support_db)

    connection.execute("""
        CREATE TABLE escalations (
            escalation_id TEXT PRIMARY KEY,
            session_id TEXT,
            condition_id TEXT,
            severity TEXT,
            queue TEXT,
            escalated_at TEXT,
            payload TEXT NOT NULL
        )
    """)

    for escalation_id, condition in (
        ("ESC-A", "ESC-03"),
        ("ESC-B", "ESC-07"),
        ("ESC-C", "ESC-03"),
    ):

        connection.execute(
            "INSERT INTO escalations VALUES (?,?,?,?,?,?,?)",
            (
                escalation_id,
                "sess-1",
                condition,
                "high",
                "priority_support",
                "2026-10-05T02:00:00+00:00",
                "{}",
            ),
        )

    connection.commit()
    connection.close()

    check(
        "the store reports the real escalation count",
        escalation_count(support_db) == 3,
        str(escalation_count(support_db)),
    )

    check(
        "the store reports escalations by condition",
        escalation_summary(support_db)
        == {"ESC-03": 2, "ESC-07": 1},
        str(escalation_summary(support_db)),
    )

    check(
        "escalations can be counted since an instant",
        escalation_count(
            support_db, since=utc(2026, 10, 5, 3, 0)
        )
        == 0,
        str(escalation_count(support_db, since=utc(2026, 10, 5, 3, 0))),
    )

    log_db = temp / "metadata.db"

    fake = FakePipeline(new=["faq.txt"])
    run = KnowledgeScheduler(fake.ports(), config())

    record = run.activate(MON_0200)

    log_run(record, db_path=log_db, support_db=support_db)

    connection = sqlite3.connect(log_db)
    connection.row_factory = sqlite3.Row

    row = connection.execute(
        "SELECT filename, status, escalations, timestamp "
        "FROM pipeline_logs"
    ).fetchone()

    connection.close()

    check("the bridge writes one log row", row is not None, str(row))
    check(
        "the log row carries the run status",
        row["status"] == STATUS_ACTIVATED,
        str(dict(row)) if row else "",
    )
    check(
        "the log row carries the real escalation count, not 0",
        row["escalations"] == 3,
        str(dict(row)) if row else "",
    )
    check(
        "the log row is timestamped at the run instant",
        row["timestamp"] == "2026-10-05 02:00:00",
        str(dict(row)) if row else "",
    )

    # A missing support store is a genuine zero, and says so.
    missing_log = temp / "missing.db"

    log_run(
        record,
        db_path=missing_log,
        support_db=temp / "not-there.db",
    )

    connection = sqlite3.connect(missing_log)

    missing_row = connection.execute(
        "SELECT escalations FROM pipeline_logs"
    ).fetchone()

    connection.close()

    check(
        "a missing support store records zero rather than failing",
        missing_row[0] == 0,
        str(missing_row),
    )


# ============================================================
# [11] THE REAL PIPELINE WIRING
# ============================================================

print("\n[11] THE REAL PIPELINE WIRING")

# Everything above injects fakes, which is what makes it fast and
# deterministic -- and also what let a broken default_ports() go
# unnoticed. These checks drive the real modules against a temp
# documents directory and a temp database.
with tempfile.TemporaryDirectory() as temp_dir:

    temp = Path(temp_dir)

    documents = temp / "documents"
    documents.mkdir()

    good = documents / "policy.txt"
    good.write_text(
        "Refunds are processed within 30 days of the request. "
        "Shipping takes 5 to 7 business days.\n",
        encoding="utf-8",
    )

    short = documents / "stub.txt"
    short.write_text("Too short.", encoding="utf-8")

    real_db = temp / "metadata.db"

    connection = sqlite3.connect(real_db)

    connection.execute("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            version INTEGER NOT NULL,
            file_hash TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            active INTEGER DEFAULT 1
        )
    """)

    connection.execute("""
        CREATE TABLE versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            version INTEGER,
            file_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()

    real = default_ports(documents_dir=documents, metadata_db=real_db)

    check("the real pipeline wiring builds", real is not None)

    detected = real.detect()

    check(
        "the real detector finds the temp documents",
        sorted(detected["new"]) == ["policy.txt", "stub.txt"],
        str(detected),
    )

    passed, message = real.quality_gate("policy.txt")

    check("the real quality gate accepts a real policy document", passed, message)

    failed, reason = real.quality_gate("stub.txt")

    check("the real quality gate rejects a too-short document", not failed, reason)

    check(
        "the real current_version is 0 before anything is saved",
        real.current_version("policy.txt") == 0,
        str(real.current_version("policy.txt")),
    )

    first = real.activate("policy.txt")

    check("the real activation returns version 1", first == 1, str(first))
    check(
        "the real activation is readable back",
        real.current_version("policy.txt") == 1,
        str(real.current_version("policy.txt")),
    )

    # The regression guard for the bug this section exists for: a
    # relative "metadata.db" would have been created next to this
    # module instead of using the database passed in.
    check(
        "the real activation writes to the database it was given",
        not (Path(__file__).resolve().parent / "metadata.db").exists(),
        str(Path(__file__).resolve().parent / "metadata.db"),
    )

    connection = sqlite3.connect(real_db)

    stored = connection.execute(
        "SELECT version, status, active FROM documents "
        "WHERE filename='policy.txt'"
    ).fetchone()

    connection.close()

    check(
        "the document row is in the temp database",
        stored == (1, "approved", 1),
        str(stored),
    )

    # Change the document and activate again, then roll back.
    good.write_text(
        "Refunds are processed within 60 days of the request. "
        "Shipping takes 3 to 5 business days.\n",
        encoding="utf-8",
    )

    second = real.activate("policy.txt")

    check("the second real activation is version 2", second == 2, str(second))

    real.rollback("policy.txt", 1)

    connection = sqlite3.connect(real_db)

    active = connection.execute(
        "SELECT version FROM documents "
        "WHERE filename='policy.txt' AND active=1"
    ).fetchall()

    connection.close()

    check(
        "the real rollback reactivates exactly the target version",
        active == [(1,)],
        str(active),
    )

    healthy, detail = real.health_check("policy.txt")

    check("the real health check passes on a present document", healthy, detail)

    healthy, detail = real.health_check("gone.txt")

    check(
        "the real health check fails on a missing document",
        not healthy,
        detail,
    )


# A run driven end to end through the real wiring, in a fresh
# environment so it starts from an empty documents directory and an
# empty database.
with tempfile.TemporaryDirectory() as temp_dir:

    temp = Path(temp_dir)

    documents = temp / "documents"
    documents.mkdir()

    (documents / "policy.txt").write_text(
        "Refunds are processed within 30 days of the request. "
        "Shipping takes 5 to 7 business days.\n",
        encoding="utf-8",
    )

    (documents / "stub.txt").write_text("Too short.", encoding="utf-8")

    real_db = temp / "metadata.db"

    connection = sqlite3.connect(real_db)

    connection.executescript("""
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            version INTEGER NOT NULL,
            file_hash TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            active INTEGER DEFAULT 1
        );
        CREATE TABLE versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            version INTEGER,
            file_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    connection.commit()
    connection.close()

    real = default_ports(documents_dir=documents, metadata_db=real_db)

    run = KnowledgeScheduler(real, config())

    record = run.activate(MON_0200)

    check(
        "the real wiring runs a clean activation",
        record.status == STATUS_ACTIVATED,
        f"{record.status}: {record.notes}",
    )

    record = run.health_check(record, MON_0200 + timedelta(minutes=5))

    check(
        "the real wiring survives its health-check window",
        record.status == STATUS_ACTIVATED and record.monitoring_complete,
        f"{record.status}: {record.notes}",
    )

    # Only the good document may have been activated; the short one is
    # quarantined by the real quality gate.
    check(
        "the real run quarantines what the gate rejects",
        record.quarantined == ["stub.txt"],
        str(record.quarantined),
    )

    check(
        "the real run activates what the gate passes",
        record.activated == ["policy.txt"],
        str(record.activated),
    )

    check(
        "the real run leaves the rejected document unversioned",
        real.current_version("stub.txt") == 0,
        str(real.current_version("stub.txt")),
    )

    check(
        "the real run records the version it activated",
        record.versions == {"policy.txt": 1},
        str(record.versions),
    )


# ============================================================
# [12] SERIALISATION
# ============================================================

print("\n[12] SERIALISATION")

fake = FakePipeline(new=["faq.txt"])
run = fake.build()

record = run.activate(MON_0200)
record = run.health_check(record, MON_0201)

payload = record.to_dict()

check("a run serialises to a dict", isinstance(payload, dict))
check("the status is serialised", payload["status"] == STATUS_ACTIVATED, str(payload["status"]))
check(
    "the start instant is serialised as ISO-8601",
    payload["started_at"] == MON_0200.isoformat(),
    str(payload["started_at"]),
)
check("the health checks are serialised", len(payload["health_checks"]) == 1, str(payload["health_checks"]))
check(
    "a health check carries its offset",
    payload["health_checks"][0]["minutes_after_activation"] == 1.0,
)
check("the activated documents are serialised", payload["activated"] == ["faq.txt"], str(payload["activated"]))
check("the versions are serialised", payload["versions"] == {"faq.txt": 1}, str(payload["versions"]))
check("a pending retry serialises as null", payload["next_retry_at"] is None)

check(
    "a run record is JSON-serialisable for the monitoring log",
    isinstance(json.dumps(payload), str),
)


print()
print("=" * 75)
print(f"PASSED: {PASS}")
print(f"FAILED: {FAIL}")
print("=" * 75)

raise SystemExit(0 if FAIL == 0 else 1)
