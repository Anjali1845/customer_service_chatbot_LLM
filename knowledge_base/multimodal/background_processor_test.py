"""
TASK 5.6 - BACKGROUND PROCESSING REGRESSION TEST

Verifies:
1. The estimate-based API stays backward compatible
   (existing signature and return keys).
2. The REAL runtime threshold: a job whose actual work
   crosses the (injectable) threshold is kept running in the
   background, the transition is recorded on the job status,
   and a notification event is emitted through the optional
   on_notify callback.
3. Jobs that finish quickly are never flagged or notified.
4. A broken on_notify callback cannot interrupt a job.
5. Real-time enforcement applies to every submitted job,
   including jobs submitted directly via manager.submit
   without any estimate.

A tiny threshold is injected so no test needs a 30-second sleep.
"""

from background_processor import (
    BACKGROUND_THRESHOLD_SECONDS,
    BackgroundJobManager,
    process_file,
    should_background,
    submit_if_long_running,
)


print("=" * 70)
print("TASK 5.6 - BACKGROUND PROCESSING REGRESSION TEST")
print("=" * 70)


# ------------------------------------------------------
# TEST 1 - ESTIMATE HEURISTIC (BACKWARD COMPATIBLE)
# ------------------------------------------------------

print("\nTEST 1 - ESTIMATE HEURISTIC")

assert should_background(10) is False
assert should_background(30) is False
assert should_background(31) is True
assert BACKGROUND_THRESHOLD_SECONDS == 30.0

print("Estimate heuristic unchanged: PASS ✅")


# ------------------------------------------------------
# TEST 2 - REAL RUNTIME CROSSES THE THRESHOLD
# ------------------------------------------------------

print("\nTEST 2 - REAL RUNTIME THRESHOLD ENFORCEMENT")

events = []

manager = BackgroundJobManager(
    max_workers=1,
    threshold_seconds=0.2,
    on_notify=events.append,
)

try:
    response = submit_if_long_running(
        manager,
        process_file,
        60,
        "slow_invoice.pdf",
        processing_time=0.6,
    )

    print("Submit response:", response)

    # Existing keys still work.
    assert response["mode"] == "background"
    assert response["job_id"] is not None
    assert response["status"] == "queued"
    assert response["estimated_seconds"] == 60

    # New key added without replacing the old ones.
    assert response["threshold_seconds"] == 0.2

    final = manager.wait_for_completion(
        response["job_id"],
        timeout=10,
    )

    print("Final status:", final)

    # The job was NOT interrupted - it kept running.
    assert final["status"] == "completed"
    assert final["result"]["processed"] is True
    assert final["duration_seconds"] >= 0.6

    # The real crossing was recorded on the job status.
    assert final["backgrounded"] is True
    assert final["backgrounded_at"] is not None
    assert final["backgrounded_after_seconds"] >= 0.2
    assert final["threshold_seconds"] == 0.2

    # A notification event was emitted to the customer.
    backgrounded = [
        event
        for event in events
        if event["type"] == "job_backgrounded"
    ]

    assert len(backgrounded) == 1
    assert backgrounded[0]["job_id"] == response["job_id"]
    assert backgrounded[0]["elapsed_seconds"] >= 0.2
    assert "background" in backgrounded[0]["message"]

    # The transition is also recorded on the job itself.
    job_events = final["notifications"]
    assert job_events[0]["type"] == "job_backgrounded"

    completed = [
        event
        for event in events
        if event["type"] == "background_job_completed"
    ]

    assert len(completed) == 1

    print("Real processing time measured: PASS ✅")
    print("Job kept running (not cancelled): PASS ✅")
    print("Transition recorded on job status: PASS ✅")
    print("Notification emitted via on_notify: PASS ✅")

finally:
    manager.shutdown()


# ------------------------------------------------------
# TEST 3 - SHORT REAL RUNTIME IS NEVER FLAGGED
# ------------------------------------------------------

print("\nTEST 3 - SHORT JOB NOT FLAGGED")

events = []

manager = BackgroundJobManager(
    max_workers=1,
    threshold_seconds=0.2,
    on_notify=events.append,
)

try:
    response = submit_if_long_running(
        manager,
        process_file,
        60,
        "tiny.png",
        processing_time=0.05,
    )

    final = manager.wait_for_completion(
        response["job_id"],
        timeout=10,
    )

    assert final["status"] == "completed"
    assert final["backgrounded"] is False
    assert final["backgrounded_at"] is None
    assert final["notifications"] == []
    assert events == []

    print("Short job completed without notification: PASS ✅")

finally:
    manager.shutdown()


# ------------------------------------------------------
# TEST 4 - NO CALLBACK / CALLBACK OPTIONALITY
# ------------------------------------------------------

print("\nTEST 4 - CALLBACK IS OPTIONAL")

manager = BackgroundJobManager(
    max_workers=1,
    threshold_seconds=0.2,
)

try:
    response = submit_if_long_running(
        manager,
        process_file,
        60,
        "no_callback.pdf",
        processing_time=0.4,
    )

    final = manager.wait_for_completion(
        response["job_id"],
        timeout=10,
    )

    # Without on_notify the job still runs, is flagged, and the
    # event is recorded on the job status itself.
    assert final["status"] == "completed"
    assert final["backgrounded"] is True
    assert final["notifications"][0][
        "type"
    ] == "job_backgrounded"

    print("Default (None) callback never crashes: PASS ✅")

finally:
    manager.shutdown()


# ------------------------------------------------------
# TEST 5 - BROKEN CALLBACK CANNOT BREAK THE JOB
# ------------------------------------------------------

print("\nTEST 5 - BROKEN CALLBACK IS ISOLATED")


def broken_callback(event):
    raise RuntimeError("notification sink failure")


manager = BackgroundJobManager(
    max_workers=1,
    threshold_seconds=0.2,
    on_notify=broken_callback,
)

try:
    response = submit_if_long_running(
        manager,
        process_file,
        60,
        "resilient.pdf",
        processing_time=0.4,
    )

    final = manager.wait_for_completion(
        response["job_id"],
        timeout=10,
    )

    assert final["status"] == "completed"
    assert final["result"]["processed"] is True
    assert final["backgrounded"] is True

    print("Broken callback never interrupts the job: PASS ✅")

finally:
    manager.shutdown()


# ------------------------------------------------------
# TEST 6 - DIRECT SUBMIT ENFORCEMENT (NO ESTIMATE)
# ------------------------------------------------------

print("\nTEST 6 - DIRECT SUBMIT REAL-TIME ENFORCEMENT")

events = []

manager = BackgroundJobManager(
    max_workers=1,
    threshold_seconds=0.2,
    on_notify=events.append,
)

try:
    # Submitted directly, bypassing the estimate API entirely.
    job_id = manager.submit(
        process_file,
        "direct.pdf",
        processing_time=0.5,
    )

    final = manager.wait_for_completion(
        job_id,
        timeout=10,
    )

    assert final["status"] == "completed"
    assert final["backgrounded"] is True

    backgrounded = [
        event
        for event in events
        if event["type"] == "job_backgrounded"
    ]

    assert len(backgrounded) == 1

    print("Real runtime enforced for direct submits: PASS ✅")

finally:
    manager.shutdown()


# ------------------------------------------------------
# TEST 7 - FOREGROUND ESTIMATE DOES NOT QUEUE
# ------------------------------------------------------

print("\nTEST 7 - FOREGROUND ESTIMATE CONTRACT")

manager = BackgroundJobManager(
    max_workers=1,
)

try:
    response = submit_if_long_running(
        manager,
        process_file,
        5,
        "small.png",
    )

    print("Response:", response)

    assert response["mode"] == "foreground"
    assert response["job_id"] is None
    assert response["status"] == "not_queued"
    assert response["estimated_seconds"] == 5
    assert response["threshold_seconds"] == 30.0

    print("Foreground contract preserved: PASS ✅")

finally:
    manager.shutdown()


# ------------------------------------------------------
# TEST 8 - FAILED BACKGROUND JOB NOTIFIES
# ------------------------------------------------------

print("\nTEST 8 - FAILED BACKGROUND JOB NOTIFICATION")

events = []


def failing_slow_task():
    # Crosses the tiny threshold before failing.
    import time

    time.sleep(0.4)
    raise RuntimeError("Simulated slow failure")


manager = BackgroundJobManager(
    max_workers=1,
    threshold_seconds=0.2,
    on_notify=events.append,
)

try:
    job_id = manager.submit(failing_slow_task)

    final = manager.wait_for_completion(
        job_id,
        timeout=10,
    )

    assert final["status"] == "failed"
    assert final["backgrounded"] is True

    event_types = [event["type"] for event in events]

    assert "job_backgrounded" in event_types
    assert "background_job_failed" in event_types

    print("Failed background job notifies the customer: PASS ✅")

finally:
    manager.shutdown()


print("\n" + "=" * 70)
print("TASK 5.6 REGRESSION TEST COMPLETE")
print("ALL 8 BACKGROUND PROCESSING TESTS PASSED ✅")
print("The 30-second rule is enforced on REAL runtime,")
print("not on caller estimates.")
print("=" * 70)
