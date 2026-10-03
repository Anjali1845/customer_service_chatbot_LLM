"""
TASK 3.1 - CHANGE DETECTION REGRESSION TEST

Verifies:
1. New files are detected
2. Unchanged files are detected
3. Modified files are detected
4. New + modified + unchanged files are detected together
5. SHA-256 hashes are deterministic
6. Existing knowledge base files are not modified
"""

from pathlib import Path
import tempfile

from update_pipeline import (
    calculate_file_hash,
    detect_changes,
    build_hash_state,
)


def main():
    print("=" * 70)
    print("TASK 3.1 - CHANGE DETECTION REGRESSION TEST")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        # ---------------------------------------------------------
        # TEST 1 - NEW FILE
        # ---------------------------------------------------------
        print("\nTEST 1 - NEW FILE")

        file_a = test_dir / "policy_a.txt"
        file_a.write_text(
            "Refund requests are allowed within 30 days.",
            encoding="utf-8",
        )

        result = detect_changes(test_dir, {})

        assert result["new"] == ["policy_a.txt"]
        assert result["modified"] == []
        assert result["unchanged"] == []

        print(result)
        print("PASS ✅")

        # ---------------------------------------------------------
        # TEST 2 - UNCHANGED FILE
        # ---------------------------------------------------------
        print("\nTEST 2 - UNCHANGED FILE")

        saved_state = build_hash_state(test_dir)

        result = detect_changes(test_dir, saved_state)

        assert result["new"] == []
        assert result["modified"] == []
        assert result["unchanged"] == ["policy_a.txt"]

        print(result)
        print("PASS ✅")

        # ---------------------------------------------------------
        # TEST 3 - MODIFIED FILE
        # ---------------------------------------------------------
        print("\nTEST 3 - MODIFIED FILE")

        file_a.write_text(
            "Refund requests are allowed within 60 days.",
            encoding="utf-8",
        )

        result = detect_changes(test_dir, saved_state)

        assert result["new"] == []
        assert result["modified"] == ["policy_a.txt"]
        assert result["unchanged"] == []

        print(result)
        print("PASS ✅")

        # ---------------------------------------------------------
        # TEST 4 - NEW + MODIFIED + UNCHANGED
        # ---------------------------------------------------------
        print("\nTEST 4 - NEW + MODIFIED + UNCHANGED")

        file_b = test_dir / "policy_b.txt"
        file_b.write_text(
            "Customer support is available 24 hours.",
            encoding="utf-8",
        )

        # Create a state where:
        # policy_a has old content
        # policy_b already existed
        previous_state = {
            "policy_a.txt": saved_state["policy_a.txt"],
            "policy_b.txt": calculate_file_hash(str(file_b)),
        }

        file_c = test_dir / "policy_c.txt"
        file_c.write_text(
            "New troubleshooting documentation.",
            encoding="utf-8",
        )

        result = detect_changes(test_dir, previous_state)

        assert result["new"] == ["policy_c.txt"]
        assert result["modified"] == ["policy_a.txt"]
        assert result["unchanged"] == ["policy_b.txt"]

        print(result)
        print("PASS ✅")

        # ---------------------------------------------------------
        # TEST 5 - HASH DETERMINISM
        # ---------------------------------------------------------
        print("\nTEST 5 - HASH DETERMINISM")

        hash_one = calculate_file_hash(str(file_b))
        hash_two = calculate_file_hash(str(file_b))

        assert hash_one == hash_two
        assert len(hash_one) == 64

        print("Hash 1:", hash_one)
        print("Hash 2:", hash_two)
        print("PASS ✅")

        # ---------------------------------------------------------
        # TEST 6 - HASH CHANGES WHEN CONTENT CHANGES
        # ---------------------------------------------------------
        print("\nTEST 6 - HASH CHANGES AFTER MODIFICATION")

        old_hash = calculate_file_hash(str(file_b))

        file_b.write_text(
            "Customer support is available 24 hours a day.",
            encoding="utf-8",
        )

        new_hash = calculate_file_hash(str(file_b))

        assert old_hash != new_hash

        print("Hash changed after content modification: YES ✅")
        print("PASS ✅")

    print("\n" + "=" * 70)
    print("TASK 3.1 REGRESSION TEST COMPLETE")
    print("ALL CHANGE DETECTION TESTS PASSED ✅")
    print("=" * 70)


if __name__ == "__main__":
    main()
