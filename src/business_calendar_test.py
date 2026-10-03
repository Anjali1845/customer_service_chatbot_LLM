"""
BUSINESS CALENDAR TEST SUITE

Covers:
- Working day vs weekend vs holiday
- Business-hours boundaries
- add_business_minutes across a weekend and a holiday
- elapsed_business_minutes over a weekend
- next_working_day / next_business_start
- exclude_weekends and exclude_holidays runtime toggles
- Timezone handling
- On-call window wrapping midnight
"""

from datetime import date, datetime, timedelta, timezone

from business_calendar import (
    BusinessCalendarConfig,
    add_business_minutes,
    business_windows,
    elapsed_business_minutes,
    is_after_hours,
    is_business_hours,
    is_on_call_hours,
    is_working_day,
    local_date,
    next_business_start,
    next_working_day,
    parse_hhmm,
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


print("=" * 75)
print("BUSINESS CALENDAR TESTS")
print("=" * 75)

# Reference week (all UTC):
#   Mon 2026-09-28 ... Fri 2026-10-02, Sat 10-03, Sun 10-04,
#   Mon 2026-10-05.
config = BusinessCalendarConfig(timezone_name="UTC")


# ============================================================
# 1. PARSING
# ============================================================

print("\n[1] HH:MM PARSING")

check("parse 09:00", parse_hhmm("09:00").hour == 9)
check("parse 18:30 minutes", parse_hhmm("18:30").minute == 30)

for bad in ("9", "0900", "25:00", "09:70", ""):
    try:
        parse_hhmm(bad)
        check(f"reject {bad!r}", False, "no ValueError raised")
    except ValueError:
        check(f"reject {bad!r}", True)


# ============================================================
# 2. WORKING DAYS
# ============================================================

print("\n[2] WORKING DAYS")

check("Monday is working", is_working_day(date(2026, 9, 28), config))
check("Friday is working", is_working_day(date(2026, 10, 2), config))
check("Saturday is not", not is_working_day(date(2026, 10, 3), config))
check("Sunday is not", not is_working_day(date(2026, 10, 4), config))

holiday_config = BusinessCalendarConfig(
    timezone_name="UTC",
    holidays=("2026-10-05",),
)

check(
    "holiday Monday is not working",
    not is_working_day(date(2026, 10, 5), holiday_config),
)

check(
    "holiday ignored when exclude_holidays=False",
    is_working_day(
        date(2026, 10, 5),
        BusinessCalendarConfig(
            timezone_name="UTC",
            holidays=("2026-10-05",),
            exclude_holidays=False,
        ),
    ),
)

seven_day = BusinessCalendarConfig(
    timezone_name="UTC", exclude_weekends=False
)

check(
    "weekend works when exclude_weekends=False",
    is_working_day(date(2026, 10, 3), seven_day),
)


# ============================================================
# 3. BUSINESS HOURS BOUNDARIES
# ============================================================

print("\n[3] BUSINESS HOURS BOUNDARIES")

check("09:00 is open", is_business_hours(utc(2026, 10, 2, 9, 0), config))
check("17:59 is open", is_business_hours(utc(2026, 10, 2, 17, 59), config))
check("18:00 is closed", not is_business_hours(utc(2026, 10, 2, 18, 0), config))
check("08:59 is closed", not is_business_hours(utc(2026, 10, 2, 8, 59), config))
check("Saturday is after hours", is_after_hours(utc(2026, 10, 3, 12, 0), config))
check("Friday 17:30 is in hours", not is_after_hours(utc(2026, 10, 2, 17, 30), config))


# ============================================================
# 4. ADD BUSINESS MINUTES
# ============================================================

print("\n[4] ADD BUSINESS MINUTES")

check(
    "Fri 17:30 + 60 -> Mon 09:30",
    add_business_minutes(utc(2026, 10, 2, 17, 30), 60, config)
    == utc(2026, 10, 5, 9, 30),
    str(add_business_minutes(utc(2026, 10, 2, 17, 30), 60, config)),
)

check(
    "Fri 17:30 + 30 -> Fri 18:00",
    add_business_minutes(utc(2026, 10, 2, 17, 30), 30, config)
    == utc(2026, 10, 2, 18, 0),
)

check(
    "Fri 17:30 + 31 -> Mon 09:01",
    add_business_minutes(utc(2026, 10, 2, 17, 30), 31, config)
    == utc(2026, 10, 5, 9, 1),
)

check(
    "weekend start rolls to Monday 09:00",
    add_business_minutes(utc(2026, 10, 3, 12, 0), 1, config)
    == utc(2026, 10, 5, 9, 1),
)

check(
    "Fri 17:30 + 60 skips holiday Monday",
    add_business_minutes(utc(2026, 10, 2, 17, 30), 60, holiday_config)
    == utc(2026, 10, 6, 9, 30),
    str(add_business_minutes(utc(2026, 10, 2, 17, 30), 60, holiday_config)),
)

check(
    "zero minutes returns the input instant",
    add_business_minutes(utc(2026, 10, 3, 12, 0), 0, config)
    == utc(2026, 10, 3, 12, 0),
)

check(
    "negative minutes returns the input instant",
    add_business_minutes(utc(2026, 10, 2, 12, 0), -30, config)
    == utc(2026, 10, 2, 12, 0),
)

check(
    "opening time + 480 -> closing time",
    add_business_minutes(utc(2026, 10, 2, 9, 0), 480, config)
    == utc(2026, 10, 2, 17, 0),
)

check(
    "opening time + 540 -> Friday closing (a full 9-hour day)",
    add_business_minutes(utc(2026, 10, 2, 9, 0), 540, config)
    == utc(2026, 10, 2, 18, 0),
    str(add_business_minutes(utc(2026, 10, 2, 9, 0), 540, config)),
)

check(
    "opening time + 541 -> Monday 09:01",
    add_business_minutes(utc(2026, 10, 2, 9, 0), 541, config)
    == utc(2026, 10, 5, 9, 1),
    str(add_business_minutes(utc(2026, 10, 2, 9, 0), 541, config)),
)


# ============================================================
# 5. ELAPSED BUSINESS MINUTES
# ============================================================

print("\n[5] ELAPSED BUSINESS MINUTES")

check(
    "same working day 09:00-12:00 is 180",
    elapsed_business_minutes(
        utc(2026, 10, 2, 9, 0), utc(2026, 10, 2, 12, 0), config
    )
    == 180.0,
)

check(
    "Friday 17:00 to Monday 10:00 is 120",
    elapsed_business_minutes(
        utc(2026, 10, 2, 17, 0), utc(2026, 10, 5, 10, 0), config
    )
    == 120.0,
    str(
        elapsed_business_minutes(
            utc(2026, 10, 2, 17, 0), utc(2026, 10, 5, 10, 0), config
        )
    ),
)

check(
    "whole weekend is 0",
    elapsed_business_minutes(
        utc(2026, 10, 3, 0, 0), utc(2026, 10, 4, 23, 59), config
    )
    == 0.0,
)

check(
    "reversed range is 0",
    elapsed_business_minutes(
        utc(2026, 10, 5, 10, 0), utc(2026, 10, 2, 10, 0), config
    )
    == 0.0,
)

check(
    "holiday contributes no minutes",
    elapsed_business_minutes(
        utc(2026, 10, 5, 0, 0), utc(2026, 10, 5, 23, 59), holiday_config
    )
    == 0.0,
)

check(
    "Saturday-open config counts the weekend",
    elapsed_business_minutes(
        utc(2026, 10, 3, 9, 0), utc(2026, 10, 3, 10, 0), seven_day
    )
    == 60.0,
)


# ============================================================
# 6. WINDOWS
# ============================================================

print("\n[6] BUSINESS WINDOWS")

windows = business_windows(
    utc(2026, 10, 2, 17, 0), utc(2026, 10, 5, 10, 0), config
)

check("two windows across a weekend", len(windows) == 2, str(windows))

check(
    "windows are clipped to the range",
    windows[0][0] == utc(2026, 10, 2, 17, 0)
    and windows[-1][1] == utc(2026, 10, 5, 10, 0),
)

check(
    "no windows in an empty range",
    business_windows(
        utc(2026, 10, 3, 0, 0), utc(2026, 10, 3, 12, 0), config
    )
    == [],
)


# ============================================================
# 7. FORWARD NAVIGATION
# ============================================================

print("\n[7] FORWARD NAVIGATION")

check(
    "next working day from Friday is Monday",
    next_working_day(utc(2026, 10, 2, 12, 0), config)
    == date(2026, 10, 5),
)

check(
    "next working day from Saturday is Monday",
    next_working_day(utc(2026, 10, 3, 12, 0), config)
    == date(2026, 10, 5),
)

check(
    "include_today accepts a working Friday",
    next_working_day(utc(2026, 10, 2, 12, 0), config, include_today=True)
    == date(2026, 10, 2),
)

check(
    "include_today skips a non-working Saturday",
    next_working_day(utc(2026, 10, 3, 12, 0), config, include_today=True)
    == date(2026, 10, 5),
)

check(
    "next business start after hours is Monday 09:00",
    next_business_start(utc(2026, 10, 2, 17, 30), config)
    == utc(2026, 10, 5, 9, 0),
)

check(
    "next business start inside hours is the next working day",
    next_business_start(utc(2026, 10, 2, 10, 0), config)
    == utc(2026, 10, 5, 9, 0),
)


# ============================================================
# 8. TIMEZONE
# ============================================================

print("\n[8] TIMEZONE")

kolkata = BusinessCalendarConfig(timezone_name="Asia/Kolkata")

# 05:00 UTC is 10:30 IST, inside business hours.
check(
    "05:00 UTC is 10:30 IST and open",
    is_business_hours(utc(2026, 10, 2, 5, 0), kolkata),
)

# 05:00 UTC is 05:00 in the UTC config, before opening.
check(
    "same instant is closed in UTC",
    not is_business_hours(utc(2026, 10, 2, 5, 0), config),
)

check(
    "local date differs around midnight UTC",
    local_date(utc(2026, 10, 1, 20, 0), kolkata) == date(2026, 10, 2),
)

# 03:30 UTC is 09:00 IST, so one business hour later is 10:00 IST = 04:30 UTC.
check(
    "03:30 UTC (09:00 IST) + 60 -> 04:30 UTC",
    add_business_minutes(utc(2026, 10, 2, 3, 30), 60, kolkata)
    == utc(2026, 10, 2, 4, 30),
    str(add_business_minutes(utc(2026, 10, 2, 3, 30), 60, kolkata)),
)

# 02:00 UTC is 07:30 IST, before opening, so the clock starts at 09:00 IST.
check(
    "03:00 UTC (08:30 IST) + 30 -> 04:00 UTC (09:30 IST)",
    add_business_minutes(utc(2026, 10, 2, 3, 0), 30, kolkata)
    == utc(2026, 10, 2, 4, 0),
    str(add_business_minutes(utc(2026, 10, 2, 3, 0), 30, kolkata)),
)


# ============================================================
# 9. ON-CALL WINDOW
# ============================================================

print("\n[9] ON-CALL WINDOW")

check("19:00 is on-call", is_on_call_hours(utc(2026, 10, 2, 19, 0), config))
check("03:00 is on-call", is_on_call_hours(utc(2026, 10, 2, 3, 0), config))
check("12:00 is not on-call", not is_on_call_hours(utc(2026, 10, 2, 12, 0), config))
check("18:00 is on-call", is_on_call_hours(utc(2026, 10, 2, 18, 0), config))
check("09:00 is not on-call", not is_on_call_hours(utc(2026, 10, 2, 9, 0), config))


# ============================================================
# 10. NAIVE DATETIME TOLERANCE
# ============================================================

print("\n[10] NAIVE DATETIME TOLERANCE")

naive = datetime(2026, 10, 2, 12, 0)

check(
    "naive datetimes are treated as UTC",
    add_business_minutes(naive, 60, config) == utc(2026, 10, 2, 13, 0),
    str(add_business_minutes(naive, 60, config)),
)

check(
    "naive and aware agree",
    elapsed_business_minutes(naive, naive + timedelta(hours=2), config)
    == elapsed_business_minutes(
        naive.replace(tzinfo=timezone.utc),
        naive.replace(tzinfo=timezone.utc) + timedelta(hours=2),
        config,
    ),
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print(f"PASSED: {PASS}")
print(f"FAILED: {FAIL}")
print("=" * 75)

raise SystemExit(0 if FAIL == 0 else 1)
