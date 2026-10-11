"""
Task 5.6 - Background Processing

Provides a lightweight local job queue for multimodal file processing.

Behavior:
- Short jobs can complete normally.
- Long-running jobs can be submitted to a background worker.
- The caller receives a job ID immediately.
- Job status can be queried.
- Jobs transition through:
    queued -> processing -> completed
                       -> failed

Real-time threshold enforcement:
- The ACTUAL running time of every submitted job is measured
  while the job runs.
- If a job keeps running past ``threshold_seconds``
  (``BACKGROUND_THRESHOLD_SECONDS`` = 30s by default), it is NOT
  cancelled: it keeps running in the background, the transition is
  recorded on the job record (``backgrounded``,
  ``backgrounded_at``, ``backgrounded_after_seconds``) and a
  notification event is emitted through the optional injectable
  ``on_notify`` callback so the application can inform the
  customer. The job's ``status`` stays ``processing`` - only the
  ``backgrounded`` flag marks the transition.
- ``estimated_seconds`` (``should_background`` /
  ``submit_if_long_running``) remains a cheap pre-submit heuristic
  for deciding whether to queue work up front; it is not the
  30-second rule itself. Real enforcement happens in
  ``BackgroundJobManager`` for every job submitted via
  ``submit`` / ``submit_if_long_running``.
"""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional


BACKGROUND_THRESHOLD_SECONDS = 30.0


class BackgroundJobManager:
    """Thread-based background job manager.

    Optional keyword arguments (all backward compatible):

    threshold_seconds:
        Real running time after which a job is flagged as
        backgrounded and a notification event is emitted.
        Defaults to ``BACKGROUND_THRESHOLD_SECONDS``.
    on_notify:
        Optional callback invoked with one notification-event
        dict per occurrence (job crossed the threshold, a
        backgrounded job finished, ...). Exceptions raised by
        the callback are swallowed so a broken notification
        sink can never interrupt the job itself. ``None`` by
        default.
    watch_poll_interval:
        How often the runtime watcher checks the elapsed time.
        Automatically reduced for small thresholds so short test
        thresholds are still detected reliably.
    """

    def __init__(
        self,
        max_workers: int = 2,
        threshold_seconds: Optional[float] = None,
        on_notify: Optional[
            Callable[[Dict[str, Any]], None]
        ] = None,
        watch_poll_interval: float = 0.5,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        if threshold_seconds is None:
            threshold_seconds = BACKGROUND_THRESHOLD_SECONDS

        threshold_seconds = float(threshold_seconds)

        if threshold_seconds <= 0:
            raise ValueError(
                "threshold_seconds must be positive"
            )

        self.executor = ThreadPoolExecutor(
            max_workers=max_workers
        )

        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

        self.threshold_seconds = threshold_seconds
        self.on_notify = on_notify
        self.watch_poll_interval = max(
            0.01,
            float(watch_poll_interval),
        )

    def _update_job(
        self,
        job_id: str,
        **updates
    ) -> None:
        """Safely update job information."""

        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id].update(updates)

    def _run_job(
        self,
        job_id: str,
        task: Callable,
        args: tuple,
        kwargs: dict
    ) -> None:
        """Execute a background job while watching its real runtime."""

        self._update_job(
            job_id,
            status="processing",
            started_at=time.time()
        )

        start = time.time()

        # Watch the job's real processing time. If it crosses
        # the threshold the job keeps running - it is only
        # flagged and a notification is emitted.
        finished = threading.Event()

        watcher = threading.Thread(
            target=self._watch_runtime,
            args=(job_id, start, finished),
            daemon=True,
        )

        watcher.start()

        try:
            result = task(*args, **kwargs)

            duration = time.time() - start

            self._update_job(
                job_id,
                status="completed",
                result=result,
                completed_at=time.time(),
                duration_seconds=duration,
            )

        except Exception as exc:
            duration = time.time() - start

            self._update_job(
                job_id,
                status="failed",
                error=str(exc),
                completed_at=time.time(),
                duration_seconds=duration,
            )

        finally:
            finished.set()
            self._notify_completion(job_id)

    def _watch_runtime(
        self,
        job_id: str,
        start: float,
        finished: threading.Event
    ) -> None:
        """
        Monitor a running job's elapsed time.

        Once the real running time crosses the threshold, the
        job is flagged as backgrounded and a notification event
        is emitted. The job itself is never interrupted.
        """

        # Poll fine-grained enough for small (test) thresholds.
        poll = min(
            self.watch_poll_interval,
            max(0.01, self.threshold_seconds / 20.0),
        )

        while not finished.wait(poll):

            elapsed = time.time() - start

            if elapsed >= self.threshold_seconds:
                self._mark_backgrounded(
                    job_id,
                    elapsed
                )
                return

    def _mark_backgrounded(
        self,
        job_id: str,
        elapsed: float
    ) -> None:
        """
        Record the foreground->background transition on the job
        and emit a notification event for the customer.
        """

        event = {
            "type": "job_backgrounded",
            "job_id": job_id,
            "elapsed_seconds": round(elapsed, 3),
            "threshold_seconds": self.threshold_seconds,
            "message": (
                "Processing your file is taking longer than "
                f"{self.threshold_seconds:g} seconds. It is still "
                "running in the background, so you can keep "
                "chatting - you will be notified when it finishes."
            ),
            "timestamp": time.time(),
        }

        with self.lock:
            job = self.jobs.get(job_id)

            # Only flag jobs that are still running. A job that
            # already finished never crossed the threshold while
            # processing.
            if (
                job is None
                or job.get("status") != "processing"
            ):
                return

            job["backgrounded"] = True
            job["backgrounded_at"] = event["timestamp"]
            job["backgrounded_after_seconds"] = (
                event["elapsed_seconds"]
            )
            job["notifications"].append(event)

        self._emit(event)

    def _notify_completion(
        self,
        job_id: str
    ) -> None:
        """
        Emit a completion notification for backgrounded jobs.

        Jobs that never crossed the threshold are not notified:
        their caller already has the result synchronously.
        """

        with self.lock:
            job = self.jobs.get(job_id)

            if job is None or not job.get("backgrounded"):
                return

            status = job.get("status")

            event = {
                "type": (
                    "background_job_completed"
                    if status == "completed"
                    else "background_job_failed"
                ),
                "job_id": job_id,
                "status": status,
                "duration_seconds": job.get(
                    "duration_seconds"
                ),
                "error": job.get("error"),
                "message": (
                    "Your file finished processing in the "
                    "background."
                    if status == "completed"
                    else "Background processing of your file "
                    "failed."
                ),
                "timestamp": time.time(),
            }

            job["notifications"].append(event)

        self._emit(event)

    def _emit(
        self,
        event: Dict[str, Any]
    ) -> None:
        """
        Forward a notification event to the injectable callback.

        The callback is optional; failures inside it are
        swallowed so notification can never break processing.
        """

        callback = self.on_notify

        if callback is None:
            return

        try:
            callback(event)
        except Exception:
            # A broken notification sink must never
            # interrupt the job itself.
            pass

    def submit(
        self,
        task: Callable,
        *args,
        **kwargs
    ) -> str:
        """
        Submit a task to the background queue.

        Returns a job ID immediately.
        """

        if not callable(task):
            raise TypeError("task must be callable")

        job_id = str(uuid.uuid4())

        with self.lock:
            self.jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "created_at": time.time(),
                "started_at": None,
                "completed_at": None,
                "duration_seconds": None,
                "result": None,
                "error": None,
                # Real runtime threshold tracking.
                "threshold_seconds": self.threshold_seconds,
                "backgrounded": False,
                "backgrounded_at": None,
                "backgrounded_after_seconds": None,
                "notifications": [],
            }

        self.executor.submit(
            self._run_job,
            job_id,
            task,
            args,
            kwargs
        )

        return job_id

    def get_status(
        self,
        job_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return a snapshot of job status."""

        with self.lock:
            job = self.jobs.get(job_id)

            if job is None:
                return None

            snapshot = dict(job)
            snapshot["notifications"] = list(
                job.get("notifications") or []
            )

            return snapshot

    def wait_for_completion(
        self,
        job_id: str,
        timeout: float = 60.0,
        poll_interval: float = 0.05
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for a job to finish.

        Useful for tests and controlled workflows.
        """

        start = time.time()

        while time.time() - start < timeout:

            status = self.get_status(job_id)

            if status is None:
                return None

            if status["status"] in {
                "completed",
                "failed"
            }:
                return status

            time.sleep(poll_interval)

        return self.get_status(job_id)

    def shutdown(self) -> None:
        """Shutdown the worker pool."""

        self.executor.shutdown(
            wait=True
        )


def process_file(
    filename: str,
    processing_time: float = 0.1
) -> Dict[str, Any]:
    """
    Simulated file-processing task.

    In the real system this represents OCR,
    extraction, comparison, validation, etc.

    The sleep is only used to simulate processing time.
    """

    if processing_time < 0:
        raise ValueError(
            "processing_time cannot be negative"
        )

    time.sleep(processing_time)

    return {
        "filename": filename,
        "processed": True,
        "message": "File processing completed."
    }


def should_background(
    estimated_seconds: float
) -> bool:
    """
    Decide whether processing should be moved
    to the background queue.

    This is only a pre-submit ESTIMATE. The real 30-second
    rule is enforced by ``BackgroundJobManager``, which
    measures the actual running time of every submitted job
    and flags/notifies when the threshold is crossed.
    """

    return (
        estimated_seconds
        > BACKGROUND_THRESHOLD_SECONDS
    )


def submit_if_long_running(
    manager: BackgroundJobManager,
    task: Callable,
    estimated_seconds: float,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Submit long-running work to the background queue.

    Short work is not automatically queued.

    The estimate only decides whether the work is QUEUED up
    front. Once queued, the manager measures the job's real
    running time: if it crosses ``manager.threshold_seconds``
    the job keeps running in the background and a notification
    event is emitted through ``manager.on_notify``.
    """

    if should_background(estimated_seconds):

        job_id = manager.submit(
            task,
            *args,
            **kwargs
        )

        return {
            "mode": "background",
            "job_id": job_id,
            "status": "queued",
            "estimated_seconds": estimated_seconds,
            "threshold_seconds": (
                manager.threshold_seconds
            ),
        }

    return {
        "mode": "foreground",
        "job_id": None,
        "status": "not_queued",
        "estimated_seconds": estimated_seconds,
        "threshold_seconds": (
            manager.threshold_seconds
        ),
    }


# --------------------------------------------------
# Self-test
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("TASK 5.6 - BACKGROUND PROCESSING SELF-TEST")
    print("=" * 70)

    manager = BackgroundJobManager(
        max_workers=2
    )

    try:

        # Test threshold.
        print("\nTEST 1 - 30 SECOND THRESHOLD")

        assert should_background(10) is False
        assert should_background(30) is False
        assert should_background(31) is True

        print("10 seconds -> foreground: YES ✅")
        print("30 seconds -> foreground: YES ✅")
        print("31 seconds -> background: YES ✅")


        # Test background submission.
        print("\nTEST 2 - BACKGROUND SUBMISSION")

        response = submit_if_long_running(
            manager,
            process_file,
            31,
            "invoice.pdf",
            processing_time=0.2
        )

        print(response)

        assert response["mode"] == "background"
        assert response["job_id"] is not None
        assert response["status"] == "queued"

        print("Job submitted to background queue: YES ✅")


        # Immediate status.
        print("\nTEST 3 - JOB STATUS")

        job_id = response["job_id"]

        status = manager.get_status(
            job_id
        )

        print(status)

        assert status is not None
        assert status["status"] in {
            "queued",
            "processing",
            "completed"
        }

        print("Job status available: YES ✅")


        # Completion.
        print("\nTEST 4 - JOB COMPLETION")

        final_status = manager.wait_for_completion(
            job_id,
            timeout=5
        )

        print(final_status)

        assert final_status["status"] == "completed"
        assert final_status["result"]["processed"] is True

        print("Background job completed: YES ✅")


        # Failure handling.
        print("\nTEST 5 - JOB FAILURE HANDLING")

        def failing_task():
            raise RuntimeError(
                "Simulated processing failure"
            )

        failure_job = manager.submit(
            failing_task
        )

        failure_status = manager.wait_for_completion(
            failure_job,
            timeout=5
        )

        print(failure_status)

        assert failure_status["status"] == "failed"
        assert "Simulated processing failure" in (
            failure_status["error"]
        )

        print("Failed job captured safely: YES ✅")


        # Foreground decision.
        print("\nTEST 6 - SHORT JOB")

        short_response = submit_if_long_running(
            manager,
            process_file,
            5,
            "small.png"
        )

        print(short_response)

        assert short_response["mode"] == "foreground"
        assert short_response["job_id"] is None
        assert short_response["threshold_seconds"] == 30.0

        print("Short job remains foreground: YES ✅")


        # Real runtime threshold.
        print("\nTEST 7 - REAL RUNTIME THRESHOLD")

        events = []

        runtime_manager = BackgroundJobManager(
            max_workers=1,
            threshold_seconds=0.2,
            on_notify=events.append,
        )

        try:
            runtime_response = submit_if_long_running(
                runtime_manager,
                process_file,
                60,
                "slow.pdf",
                processing_time=0.5
            )

            assert runtime_response["mode"] == "background"
            assert runtime_response["threshold_seconds"] == 0.2

            runtime_status = (
                runtime_manager.wait_for_completion(
                    runtime_response["job_id"],
                    timeout=10
                )
            )

            print(runtime_status)

            # The job was NOT interrupted - it kept running.
            assert runtime_status["status"] == "completed"
            assert runtime_status["result"]["processed"] is True
            assert (
                runtime_status["duration_seconds"] >= 0.5
            )

            # The real crossing was recorded on the job.
            assert runtime_status["backgrounded"] is True
            assert (
                runtime_status["backgrounded_after_seconds"]
                >= 0.2
            )

            # A notification event was emitted to the customer.
            backgrounded_events = [
                event
                for event in events
                if event["type"] == "job_backgrounded"
            ]

            assert len(backgrounded_events) == 1
            assert (
                runtime_status["notifications"][0]["type"]
                == "job_backgrounded"
            )

            print("Real runtime measured: YES ✅")
            print("Job kept running in background: YES ✅")
            print("Transition recorded on job: YES ✅")
            print("Customer notification emitted: YES ✅")

        finally:
            runtime_manager.shutdown()


        print("\n" + "=" * 70)
        print("TASK 5.6 SELF-TEST COMPLETE")
        print("BACKGROUND PROCESSING: PASS ✅")
        print("=" * 70)

    finally:
        manager.shutdown()
