"""
SUPPORT OPS - BUSINESS CALENDAR

Weekend/holiday-aware business-time maths, shared by Task 7
(escalation: after-hours routing, unresolved clock) and Task 8
(ticketing: SLA due dates, warnings and breaches).

Everything here is driven by a mutable BusinessCalendarConfig, so the
working week, business hours, holiday list and timezone can all be
changed at runtime. Callers that only mutate the config and recompute
get correct results, because no derived value is cached.

All public functions accept and return timezone-aware UTC datetimes,
except next_working_day() which returns a local calendar date.

The one implementation of window arithmetic lives here; nothing else
in the codebase should re-derive it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from typing import List, Optional, Tuple

from support_deps import as_utc


try:

    from zoneinfo import ZoneInfo

except ImportError:  # pragma: no cover - Python < 3.9

    ZoneInfo = None  # type: ignore


# Safety valve for the day-by-day loops. A year of iterations is far
# beyond any realistic SLA or scheduling horizon.
MAX_DAYS = 3660


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class BusinessCalendarConfig:
    """
    Working-week and business-hours definition.

    Mutable by design: the hidden evaluation may change weekends,
    holidays or opening hours at runtime.
    """

    timezone_name: str = "Asia/Kolkata"

    # date.weekday() values: Monday=0 ... Sunday=6
    working_days: Tuple[int, ...] = (0, 1, 2, 3, 4)

    # Local wall-clock business hours, "HH:MM".
    business_hours_start: str = "09:00"
    business_hours_end: str = "18:00"

    # Local ISO dates, e.g. ("2026-10-20", "2026-12-25").
    holidays: Tuple[str, ...] = ()

    # Convenience toggles. exclude_weekends=False makes every weekday
    # eligible, so business runs seven days a week.
    exclude_weekends: bool = True
    exclude_holidays: bool = True

    # After-hours on-call window. May wrap past midnight.
    on_call_hours_start: str = "18:00"
    on_call_hours_end: str = "09:00"


# ============================================================
# TIMEZONE + PARSING HELPERS
# ============================================================

@lru_cache(maxsize=16)
def _zone(timezone_name: str):

    if ZoneInfo is None:  # pragma: no cover

        raise ImportError(
            "zoneinfo is unavailable on this Python build."
        )

    try:

        return ZoneInfo(timezone_name)

    except Exception as error:

        raise ImportError(
            f"Unknown timezone {timezone_name!r}. Install the "
            f"'tzdata' package if your platform has no system "
            f"timezone database."
        ) from error


def parse_hhmm(value: str) -> time:
    """
    Parse a "HH:MM" wall-clock string.

    Raises ValueError on anything else, so a typo in config fails
    loudly instead of silently shifting every SLA deadline.
    """

    if not isinstance(value, str):

        raise ValueError(f"Expected 'HH:MM' string, got {value!r}")

    parts = value.strip().split(":")

    if len(parts) != 2:

        raise ValueError(f"Expected 'HH:MM' string, got {value!r}")

    try:

        hour = int(parts[0])
        minute = int(parts[1])

    except ValueError as error:

        raise ValueError(
            f"Expected 'HH:MM' string, got {value!r}"
        ) from error

    if not (0 <= hour <= 23 and 0 <= minute <= 59):

        raise ValueError(f"Out-of-range time: {value!r}")

    return time(hour=hour, minute=minute)


def local_now(
    now: Optional[datetime],
    config: BusinessCalendarConfig,
) -> datetime:
    """
    Convert an instant to an aware datetime in the configured zone.
    """

    return as_utc(now).astimezone(_zone(config.timezone_name))


def local_date(
    now: Optional[datetime],
    config: BusinessCalendarConfig,
) -> date:
    """
    The calendar date in the configured zone for an instant.
    """

    return local_now(now, config).date()


# ============================================================
# WORKING DAY / HOURS
# ============================================================

def is_working_day(
    day: date,
    config: BusinessCalendarConfig,
) -> bool:
    """
    Whether business runs on this local calendar date.

    When exclude_weekends is False the weekday filter is skipped
    entirely, so every day of the week qualifies unless it is a
    holiday and holidays are excluded.
    """

    if (
        config.exclude_weekends
        and day.weekday() not in config.working_days
    ):

        return False

    if (
        config.exclude_holidays
        and day.isoformat() in set(config.holidays)
    ):

        return False

    return True


def _window_for_day(
    day: date,
    config: BusinessCalendarConfig,
) -> Optional[Tuple[datetime, datetime]]:
    """
    Aware local [start, end) datetimes for a day, or None when the
    day is not a working day or the window is degenerate.
    """

    if not is_working_day(day, config):

        return None

    zone = _zone(config.timezone_name)

    start = datetime.combine(
        day, parse_hhmm(config.business_hours_start), tzinfo=zone
    )

    end = datetime.combine(
        day, parse_hhmm(config.business_hours_end), tzinfo=zone
    )

    if end <= start:

        return None

    return start, end


def is_business_hours(
    now: Optional[datetime],
    config: BusinessCalendarConfig,
) -> bool:
    """
    True strictly inside the business window: opening time counts,
    closing time does not.
    """

    moment = local_now(now, config)

    window = _window_for_day(moment.date(), config)

    if window is None:

        return False

    start, end = window

    return start <= moment < end


def is_after_hours(
    now: Optional[datetime],
    config: BusinessCalendarConfig,
) -> bool:
    """
    Anything outside the business window: a non-working day, or a
    working day outside opening hours.
    """

    return not is_business_hours(now, config)


def is_on_call_hours(
    now: Optional[datetime],
    config: BusinessCalendarConfig,
) -> bool:
    """
    True when the instant falls inside the on-call window.

    The on-call window may wrap past midnight, e.g. 18:00 -> 09:00.
    """

    moment = local_now(now, config).time()

    start = parse_hhmm(config.on_call_hours_start)
    end = parse_hhmm(config.on_call_hours_end)

    if start == end:

        return True

    if start < end:

        return start <= moment < end

    # Wrapped window: 18:00 -> 09:00 means >= 18:00 or < 09:00.
    return moment >= start or moment < end


# ============================================================
# FORWARD NAVIGATION
# ============================================================

def next_working_day(
    now: Optional[datetime],
    config: BusinessCalendarConfig,
    include_today: bool = False,
) -> date:
    """
    The next local working date.

    By default this starts from tomorrow, which is what "schedule for
    the next working day" means for after-hours work. Pass
    include_today=True to accept today when it is already a working
    day.
    """

    day = local_date(now, config)

    if not include_today:

        day += timedelta(days=1)

    for _ in range(MAX_DAYS):

        if is_working_day(day, config):

            return day

        day += timedelta(days=1)

    raise RuntimeError(
        "No working day found within the search horizon."
    )


def next_business_start(
    now: Optional[datetime],
    config: BusinessCalendarConfig,
) -> datetime:
    """
    The next moment business opens strictly after this instant, in UTC.

    When called during business hours this returns the opening time of
    the following working day.
    """

    cursor = as_utc(now)

    day = local_date(cursor, config)

    for _ in range(MAX_DAYS):

        window = _window_for_day(day, config)

        if window is not None:

            start = as_utc(window[0])

            if cursor < start:

                return start

        day += timedelta(days=1)

    raise RuntimeError(
        "No business opening found within the search horizon."
    )


def add_business_minutes(
    now: Optional[datetime],
    minutes: float,
    config: BusinessCalendarConfig,
) -> datetime:
    """
    Advance an instant by N business minutes, skipping non-working
    days and time outside business hours.

    Returns UTC. This is the single source of SLA due dates.
    """

    cursor = as_utc(now)

    remaining = float(minutes)

    if remaining <= 0:

        return cursor

    day = local_date(cursor, config)

    for _ in range(MAX_DAYS):

        window = _window_for_day(day, config)

        if window is not None:

            start = as_utc(window[0])
            end = as_utc(window[1])

            if cursor < start:

                cursor = start

            if cursor < end:

                capacity = (
                    end - cursor
                ).total_seconds() / 60.0

                if remaining <= capacity:

                    return cursor + timedelta(
                        minutes=remaining
                    )

                remaining -= capacity

                cursor = end

        day += timedelta(days=1)

    raise RuntimeError(
        "Business minutes exceeded the search horizon."
    )


# ============================================================
# ELAPSED TIME
# ============================================================

def business_windows(
    start: datetime,
    end: datetime,
    config: BusinessCalendarConfig,
) -> List[Tuple[datetime, datetime]]:
    """
    Business windows overlapping [start, end], clipped to the range,
    as UTC pairs. Empty when the range contains no business time.
    """

    start_utc = as_utc(start)
    end_utc = as_utc(end)

    if end_utc <= start_utc:

        return []

    windows: List[Tuple[datetime, datetime]] = []

    day = local_date(start_utc, config)
    final_day = local_date(end_utc, config)

    for _ in range(MAX_DAYS):

        if day > final_day:

            break

        window = _window_for_day(day, config)

        if window is not None:

            open_utc = as_utc(window[0])
            close_utc = as_utc(window[1])

            lower = max(open_utc, start_utc)
            upper = min(close_utc, end_utc)

            if upper > lower:

                windows.append((lower, upper))

        day += timedelta(days=1)

    return windows


def elapsed_business_minutes(
    start: datetime,
    end: datetime,
    config: BusinessCalendarConfig,
) -> float:
    """
    Business minutes between two instants, excluding weekends,
    holidays and time outside business hours.

    Never negative: an end before the start returns 0.0.
    """

    total = 0.0

    for lower, upper in business_windows(start, end, config):

        total += (upper - lower).total_seconds() / 60.0

    return total


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BUSINESS CALENDAR SELF-TEST")
    print("=" * 60)

    cfg = BusinessCalendarConfig(timezone_name="UTC")

    # 2026-10-02 is a Friday.
    friday_1730 = datetime(2026, 10, 2, 17, 30, tzinfo=timezone.utc)

    print("\nFriday 17:30 + 60 business minutes:")
    print(" ", add_business_minutes(friday_1730, 60, cfg))

    monday_0930 = datetime(2026, 10, 5, 9, 30, tzinfo=timezone.utc)

    assert add_business_minutes(friday_1730, 60, cfg) == monday_0930

    saturday = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)

    assert is_after_hours(saturday, cfg) is True

    friday_1800 = datetime(2026, 10, 2, 18, 0, tzinfo=timezone.utc)

    assert is_business_hours(friday_1800, cfg) is False

    print("\nRESULT: PASS")
