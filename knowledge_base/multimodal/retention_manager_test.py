from pathlib import Path
import tempfile

from retention_manager import (
    RetentionManager,
    create_test_file,
)


print("=" * 70)
print("TASK 5.7 - RETENTION & SECURE DELETION REGRESSION TEST")
print("=" * 70)


with tempfile.TemporaryDirectory() as temp_dir:

    storage = Path(temp_dir) / "managed_storage"

    manager = RetentionManager(
        storage,
        retention_seconds=10
    )


    # --------------------------------------------------
    # TEST 1 - RECENT FILE MUST SURVIVE
    # --------------------------------------------------
    print("\nTEST 1 - RECENT FILE PRESERVATION")

    recent = create_test_file(
        storage / "recent.pdf",
        content="RECENT DATA",
        age_seconds=2
    )

    result = manager.delete_file(
        str(recent)
    )

    print(result)

    assert result["deleted"] is False
    assert recent.exists()
    assert recent.read_text() == "RECENT DATA"

    print("Recent file preserved: YES ✅")
    print("RESULT: PASS ✅")


    # --------------------------------------------------
    # TEST 2 - EXPIRED FILE MUST BE DELETED
    # --------------------------------------------------
    print("\nTEST 2 - EXPIRED FILE DELETION")

    expired = create_test_file(
        storage / "expired.pdf",
        content="OLD DATA",
        age_seconds=20
    )

    result = manager.delete_file(
        str(expired)
    )

    print(result)

    assert result["deleted"] is True
    assert not expired.exists()

    print("Expired file deleted: YES ✅")
    print("RESULT: PASS ✅")


    # --------------------------------------------------
    # TEST 3 - EXPIRATION DETECTION
    # --------------------------------------------------
    print("\nTEST 3 - EXPIRATION DETECTION")

    old_file = create_test_file(
        storage / "old.txt",
        age_seconds=20
    )

    new_file = create_test_file(
        storage / "new.txt",
        age_seconds=1
    )

    assert manager.is_expired(
        str(old_file)
    ) is True

    assert manager.is_expired(
        str(new_file)
    ) is False

    print("Expired file identified: YES ✅")
    print("Active file identified: YES ✅")
    print("RESULT: PASS ✅")


    # --------------------------------------------------
    # TEST 4 - BULK CLEANUP
    # --------------------------------------------------
    print("\nTEST 4 - BULK CLEANUP")

    create_test_file(
        storage / "cleanup_old_1.pdf",
        age_seconds=20
    )

    create_test_file(
        storage / "cleanup_old_2.pdf",
        age_seconds=30
    )

    cleanup_recent = create_test_file(
        storage / "cleanup_recent.pdf",
        age_seconds=1
    )

    result = manager.cleanup_expired()

    print(result)

    assert result["deleted_count"] >= 2
    assert cleanup_recent.exists()

    print("Expired files cleaned: YES ✅")
    print("Recent files preserved: YES ✅")
    print("RESULT: PASS ✅")


    # --------------------------------------------------
    # TEST 5 - OUTSIDE STORAGE PROTECTION
    # --------------------------------------------------
    print("\nTEST 5 - OUTSIDE STORAGE PROTECTION")

    outside = Path(temp_dir) / "outside_sensitive.txt"

    create_test_file(
        outside,
        content="SENSITIVE DATA",
        age_seconds=100
    )

    result = manager.delete_file(
        str(outside)
    )

    print(result)

    assert result["deleted"] is False
    assert outside.exists()
    assert outside.read_text() == "SENSITIVE DATA"

    print("Outside file protected: YES ✅")
    print("Sensitive outside file preserved: YES ✅")
    print("RESULT: PASS ✅")


    # --------------------------------------------------
    # TEST 6 - PATH TRAVERSAL
    # --------------------------------------------------
    print("\nTEST 6 - PATH TRAVERSAL PROTECTION")

    traversal = (
        storage
        / ".."
        / "outside_sensitive.txt"
    )

    result = manager.delete_file(
        str(traversal)
    )

    print(result)

    assert result["deleted"] is False
    assert outside.exists()

    print("Path traversal blocked: YES ✅")
    print("RESULT: PASS ✅")


    # --------------------------------------------------
    # TEST 7 - CONTENT PRESERVATION
    # --------------------------------------------------
    print("\nTEST 7 - CONTENT PRESERVATION")

    protected = create_test_file(
        storage / "protected.pdf",
        content="ORIGINAL CUSTOMER EVIDENCE",
        age_seconds=1
    )

    before = protected.read_text(
        encoding="utf-8"
    )

    # Attempt cleanup while file is still active.
    manager.cleanup_expired()

    after = protected.read_text(
        encoding="utf-8"
    )

    assert before == "ORIGINAL CUSTOMER EVIDENCE"
    assert after == before
    assert protected.exists()

    print("Original content preserved: YES ✅")
    print("Active evidence not deleted: YES ✅")
    print("RESULT: PASS ✅")


    # --------------------------------------------------
    # TEST 8 - DELETION AUDIT
    # --------------------------------------------------
    print("\nTEST 8 - DELETION RESULT / AUDIT")

    audit_file = create_test_file(
        storage / "audit_old.pdf",
        age_seconds=20
    )

    result = manager.delete_file(
        str(audit_file)
    )

    print(result)

    assert result["deleted"] is True
    assert result["reason"] == "Expired file deleted."
    assert "path" in result

    print("Deletion result recorded: YES ✅")
    print("Deletion reason recorded: YES ✅")
    print("Deleted path recorded: YES ✅")
    print("RESULT: PASS ✅")


print("\n" + "=" * 70)
print("TASK 5.7 REGRESSION TEST COMPLETE")
print("=" * 70)
print("ALL 8 RETENTION & DELETION TESTS PASSED ✅")
print("Expired files are removed safely.")
print("Active files are preserved.")
print("Outside files cannot be deleted.")
print("=" * 70)
