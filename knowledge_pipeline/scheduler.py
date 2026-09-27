import time
from datetime import datetime

# Configurable schedule
UPDATE_TIME = "18:00"          # 6 PM
RETRY_MINUTES = [15, 30, 60]   # Retry intervals

def check_for_updates():
    print("🔍 Checking for new or modified documents...")
    # Later this will call duplicate_detector.py
    return False  # Simulating no new files

def activate_update():
    print("🚀 Activating approved knowledge base update...")

def retry_failed_update():
    print("⚠️ Update failed.")
    for minute in RETRY_MINUTES:
        print(f"Retry scheduled after {minute} minutes")

def scheduler():
    now = datetime.now().strftime("%H:%M")
    print(f"Current Time: {now}")

    if now == UPDATE_TIME:
        print("🟢 Maintenance window started")

        success = check_for_updates()

        if success:
            activate_update()
        else:
            retry_failed_update()
    else:
        print("⏳ Waiting for scheduled update window...")

if __name__ == "__main__":
    scheduler()