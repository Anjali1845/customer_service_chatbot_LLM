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
"""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional


BACKGROUND_THRESHOLD_SECONDS = 30.0


class BackgroundJobManager:
    """Thread-based background job manager."""

    def __init__(self, max_workers: int = 2):
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")

        self.executor = ThreadPoolExecutor(
            max_workers=max_workers
        )

        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

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
        """Execute a background job."""

        self._update_job(
            job_id,
            status="processing",
            started_at=time.time()
        )

        start = time.time()

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

            return dict(job)

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
        }

    return {
        "mode": "foreground",
        "job_id": None,
        "status": "not_queued",
        "estimated_seconds": estimated_seconds,
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

        print("Short job remains foreground: YES ✅")


        print("\n" + "=" * 70)
        print("TASK 5.6 SELF-TEST COMPLETE")
        print("BACKGROUND PROCESSING: PASS ✅")
        print("=" * 70)

    finally:
        manager.shutdown()
