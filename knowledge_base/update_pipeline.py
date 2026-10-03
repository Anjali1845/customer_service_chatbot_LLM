"""
TASK 3.1 - KNOWLEDGE BASE CHANGE DETECTION

Detects:
- NEW documents
- MODIFIED documents
- UNCHANGED documents

Uses SHA-256 content hashes.

This module does NOT modify the existing knowledge base.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List


BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"

STATE_FILE = BASE_DIR / "metadata" / "content_hashes.json"


def calculate_file_hash(
    file_path: str
) -> str:
    """Return SHA-256 hash of a file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Not a regular file: {path}"
        )

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def load_hash_state() -> Dict[str, str]:
    """Load previously recorded content hashes."""

    if not STATE_FILE.exists():
        return {}

    with open(
        STATE_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "content_hashes.json must contain an object."
        )

    return data


def save_hash_state(
    state: Dict[str, str]
) -> None:
    """Persist content hashes."""

    STATE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        STATE_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            state,
            file,
            indent=2,
            sort_keys=True
        )


def detect_changes(
    documents_dir: Path = DOCUMENTS_DIR,
    previous_state: Dict[str, str] | None = None
) -> Dict[str, List[str]]:
    """
    Compare current files against previous hashes.

    Returns:
        {
            "new": [...],
            "modified": [...],
            "unchanged": [...]
        }
    """

    if previous_state is None:
        previous_state = load_hash_state()

    new_files = []
    modified_files = []
    unchanged_files = []

    current_state = {}

    for path in sorted(documents_dir.iterdir()):

        if not path.is_file():
            continue

        # Ignore hidden/system files.
        if path.name.startswith("."):
            continue

        file_hash = calculate_file_hash(
            str(path)
        )

        current_state[path.name] = file_hash

        previous_hash = previous_state.get(
            path.name
        )

        if previous_hash is None:
            new_files.append(path.name)

        elif previous_hash != file_hash:
            modified_files.append(path.name)

        else:
            unchanged_files.append(path.name)

    return {
        "new": new_files,
        "modified": modified_files,
        "unchanged": unchanged_files,
    }


def build_hash_state(
    documents_dir: Path = DOCUMENTS_DIR
) -> Dict[str, str]:
    """Build current hash state without saving it."""

    state = {}

    for path in sorted(documents_dir.iterdir()):

        if not path.is_file():
            continue

        if path.name.startswith("."):
            continue

        state[path.name] = calculate_file_hash(
            str(path)
        )

    return state


# --------------------------------------------------
# Self-test
# --------------------------------------------------

if __name__ == "__main__":

    import tempfile

    print("=" * 70)
    print("TASK 3.1 - CHANGE DETECTION SELF-TEST")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:

        test_dir = Path(temp_dir)

        # Initial files.
        file_a = test_dir / "policy_a.txt"
        file_b = test_dir / "policy_b.txt"

        file_a.write_text(
            "Refund within 30 days.",
            encoding="utf-8"
        )

        file_b.write_text(
            "Support is available 24 hours.",
            encoding="utf-8"
        )

        # No previous state.
        first_state = {}

        result = detect_changes(
            test_dir,
            first_state
        )

        print("\nTEST 1 - NEW FILES")
        print(result)

        assert set(result["new"]) == {
            "policy_a.txt",
            "policy_b.txt"
        }

        assert result["modified"] == []
        assert result["unchanged"] == []

        print("New files detected: YES ✅")


        # Build state after first scan.
        saved_state = build_hash_state(
            test_dir
        )

        # Second scan without changes.
        result = detect_changes(
            test_dir,
            saved_state
        )

        print("\nTEST 2 - UNCHANGED FILES")
        print(result)

        assert result["new"] == []
        assert result["modified"] == []
        assert set(result["unchanged"]) == {
            "policy_a.txt",
            "policy_b.txt"
        }

        print("Unchanged files detected: YES ✅")


        # Modify one file.
        file_a.write_text(
            "Refund within 60 days.",
            encoding="utf-8"
        )

        result = detect_changes(
            test_dir,
            saved_state
        )

        print("\nTEST 3 - MODIFIED FILE")
        print(result)

        assert result["new"] == []
        assert result["modified"] == [
            "policy_a.txt"
        ]
        assert result["unchanged"] == [
            "policy_b.txt"
        ]

        print("Modified file detected: YES ✅")


        # Add a new file.
        file_c = test_dir / "policy_c.txt"

        file_c.write_text(
            "New troubleshooting guide.",
            encoding="utf-8"
        )

        result = detect_changes(
            test_dir,
            saved_state
        )

        print("\nTEST 4 - NEW + MODIFIED + UNCHANGED")
        print(result)

        assert result["new"] == [
            "policy_c.txt"
        ]

        assert result["modified"] == [
            "policy_a.txt"
        ]

        assert result["unchanged"] == [
            "policy_b.txt"
        ]

        print("All change categories detected: YES ✅")


        # Hash consistency.
        hash_one = calculate_file_hash(
            str(file_b)
        )

        hash_two = calculate_file_hash(
            str(file_b)
        )

        print("\nTEST 5 - HASH CONSISTENCY")

        assert hash_one == hash_two
        assert len(hash_one) == 64

        print("SHA-256 hash generated: YES ✅")
        print("Hash is deterministic: YES ✅")


    print("\n" + "=" * 70)
    print("TASK 3.1 SELF-TEST COMPLETE")
    print("CHANGE DETECTION: PASS ✅")
    print("=" * 70)
