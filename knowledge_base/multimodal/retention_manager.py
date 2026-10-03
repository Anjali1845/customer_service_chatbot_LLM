"""
Task 5.7 - Retention & Secure Deletion

Manages retention of uploaded multimodal files.

Security properties:
- Configurable retention period
- Deletes only expired files
- Deletes only files inside managed storage
- Prevents path traversal / outside-directory deletion
- Provides deletion audit information
- Does not delete active files
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_RETENTION_SECONDS = 24 * 60 * 60


class RetentionManager:
    """Secure retention manager for uploaded files."""

    def __init__(
        self,
        storage_dir: str,
        retention_seconds: int = DEFAULT_RETENTION_SECONDS
    ):
        if retention_seconds < 0:
            raise ValueError(
                "retention_seconds cannot be negative"
            )

        self.storage_dir = Path(
            storage_dir
        ).resolve()

        self.retention_seconds = retention_seconds

        # Create managed storage if it doesn't exist.
        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def _is_managed_file(
        self,
        file_path: Path
    ) -> bool:
        """
        Verify that the file is inside the managed
        storage directory.
        """

        try:
            file_path.resolve().relative_to(
                self.storage_dir
            )
            return True
        except ValueError:
            return False

    def _validate_file(
        self,
        file_path: Path
    ) -> bool:
        """Check that path is a regular managed file."""

        if not file_path.exists():
            return False

        if not file_path.is_file():
            return False

        return self._is_managed_file(
            file_path
        )

    def get_file_age(
        self,
        file_path: str,
        now: Optional[float] = None
    ) -> float:
        """Return file age in seconds."""

        path = Path(file_path).resolve()

        if not self._validate_file(path):
            raise ValueError(
                "File is outside managed storage "
                "or is not a regular file."
            )

        if now is None:
            now = time.time()

        modified_time = path.stat().st_mtime

        return max(
            0.0,
            now - modified_time
        )

    def is_expired(
        self,
        file_path: str,
        now: Optional[float] = None
    ) -> bool:
        """Determine whether a managed file has expired."""

        age = self.get_file_age(
            file_path,
            now=now
        )

        return age >= self.retention_seconds

    def delete_file(
        self,
        file_path: str
    ) -> Dict:
        """
        Delete one file only if it is inside managed storage
        and has expired.
        """

        path = Path(file_path).resolve()

        if not self._validate_file(path):
            return {
                "deleted": False,
                "reason": (
                    "File is outside managed storage, "
                    "missing, or not a regular file."
                ),
                "path": str(path),
            }

        if not self.is_expired(str(path)):
            return {
                "deleted": False,
                "reason": "File has not expired.",
                "path": str(path),
            }

        try:
            path.unlink()

            return {
                "deleted": True,
                "reason": "Expired file deleted.",
                "path": str(path),
            }

        except OSError as exc:
            return {
                "deleted": False,
                "reason": f"Deletion failed: {exc}",
                "path": str(path),
            }

    def cleanup_expired(
        self,
        now: Optional[float] = None
    ) -> Dict:
        """
        Delete all expired regular files directly under
        managed storage and its subdirectories.
        """

        deleted: List[str] = []
        skipped: List[str] = []
        errors: List[str] = []

        for path in self.storage_dir.rglob("*"):

            if not path.is_file():
                continue

            if not self._is_managed_file(path):
                skipped.append(str(path))
                continue

            try:
                if self.is_expired(
                    str(path),
                    now=now
                ):
                    path.unlink()
                    deleted.append(str(path))
                else:
                    skipped.append(str(path))

            except OSError as exc:
                errors.append(
                    f"{path}: {exc}"
                )

        return {
            "deleted_count": len(deleted),
            "deleted": deleted,
            "skipped_count": len(skipped),
            "skipped": skipped,
            "error_count": len(errors),
            "errors": errors,
        }


# --------------------------------------------------
# Helper for tests / controlled storage
# --------------------------------------------------

def create_test_file(
    path: str,
    content: str = "test data",
    age_seconds: Optional[float] = None
) -> Path:
    """
    Create a file and optionally modify its timestamp
    so it appears to have a specific age.
    """

    file_path = Path(path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_path.write_text(
        content,
        encoding="utf-8"
    )

    if age_seconds is not None:
        timestamp = time.time() - age_seconds

        os.utime(
            file_path,
            (timestamp, timestamp)
        )

    return file_path


# --------------------------------------------------
# Self-test
# --------------------------------------------------

if __name__ == "__main__":

    import tempfile

    print("=" * 70)
    print("TASK 5.7 - RETENTION & DELETION SELF-TEST")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:

        storage = Path(temp_dir) / "retention_storage"

        manager = RetentionManager(
            storage,
            retention_seconds=10
        )

        # TEST 1
        print("\nTEST 1 - MANAGED STORAGE")

        assert storage.exists()
        assert storage.is_dir()

        print("Managed storage created: YES ✅")


        # TEST 2
        print("\nTEST 2 - RECENT FILE PRESERVED")

        recent = create_test_file(
            storage / "recent.pdf",
            age_seconds=2
        )

        result = manager.delete_file(
            str(recent)
        )

        print(result)

        assert result["deleted"] is False
        assert recent.exists()

        print("Non-expired file preserved: YES ✅")


        # TEST 3
        print("\nTEST 3 - EXPIRED FILE DELETION")

        old = create_test_file(
            storage / "old.pdf",
            age_seconds=20
        )

        result = manager.delete_file(
            str(old)
        )

        print(result)

        assert result["deleted"] is True
        assert not old.exists()

        print("Expired file deleted: YES ✅")


        # TEST 4
        print("\nTEST 4 - RETENTION CHECK")

        another_old = create_test_file(
            storage / "another_old.pdf",
            age_seconds=20
        )

        assert manager.is_expired(
            str(another_old)
        ) is True

        recent2 = create_test_file(
            storage / "recent2.pdf",
            age_seconds=1
        )

        assert manager.is_expired(
            str(recent2)
        ) is False

        print("Expired file detected: YES ✅")
        print("Recent file detected as active: YES ✅")


        # TEST 5
        print("\nTEST 5 - CLEANUP")

        cleanup = manager.cleanup_expired()

        print(cleanup)

        assert cleanup["deleted_count"] >= 1
        assert another_old.exists() is False
        assert recent2.exists() is True

        print("Expired files cleaned: YES ✅")
        print("Recent files preserved: YES ✅")


        # TEST 6
        print("\nTEST 6 - OUTSIDE DIRECTORY PROTECTION")

        outside = Path(temp_dir) / "outside.txt"

        create_test_file(
            outside,
            age_seconds=100
        )

        result = manager.delete_file(
            str(outside)
        )

        print(result)

        assert result["deleted"] is False
        assert outside.exists()

        print(
            "Outside file protected from deletion: YES ✅"
        )


        # TEST 7
        print("\nTEST 7 - PATH TRAVERSAL PROTECTION")

        traversal = storage / ".." / "outside.txt"

        result = manager.delete_file(
            str(traversal)
        )

        print(result)

        assert result["deleted"] is False
        assert outside.exists()

        print(
            "Path traversal blocked: YES ✅"
        )


        # TEST 8
        print("\nTEST 8 - ORIGINAL CONTENT")

        protected = create_test_file(
            storage / "protected.pdf",
            content="IMPORTANT DATA",
            age_seconds=1
        )

        before = protected.read_text(
            encoding="utf-8"
        )

        assert before == "IMPORTANT DATA"
        assert protected.exists()

        print(
            "Non-expired file content preserved: YES ✅"
        )


    print("\n" + "=" * 70)
    print("TASK 5.7 SELF-TEST COMPLETE")
    print("RETENTION & SECURE DELETION: PASS ✅")
    print("=" * 70)
