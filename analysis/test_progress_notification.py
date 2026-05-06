#!/usr/bin/env python3
"""
Test progress notification updates to verify that updates replace existing notifications.
"""

import time
import notifications

def test_progress_updates():
    """Test that progress updates replace the same notification."""
    
    if not notifications.is_enabled():
        print("⚠️  Notifications are NOT enabled!")
        print("Set NOTIFY_METHOD and related environment variables.")
        return
    
    print("✓ Notifications are enabled!")
    print(f"  Method: {notifications._notifier.method}")
    print("\nTesting progress notification updates...")
    print("You should see ONE notification that updates 3 times, not 3 separate notifications.\n")
    
    # Create initial progress notification
    job_id = notifications.create_progress_notification(
        job_id="test_job_123",
        title="Test Progress",
        initial_message="Starting test... (0/3)"
    )
    print("1. Sent initial notification")
    time.sleep(3)
    
    # Update 1
    notifications.update_progress_notification(
        job_id=job_id,
        message="Test in progress... (1/3)",
        title="Test Progress"
    )
    print("2. Sent update 1")
    time.sleep(3)
    
    # Update 2
    notifications.update_progress_notification(
        job_id=job_id,
        message="Almost done... (2/3)",
        title="Test Progress"
    )
    print("3. Sent update 2")
    time.sleep(3)
    
    # Final completion
    notifications.complete_progress_notification(
        job_id=job_id,
        final_message="Test complete! (3/3)",
        title="✓ Test Complete"
    )
    print("4. Sent completion notification")
    
    print("\n" + "=" * 60)
    print("✓ Test complete!")
    print("\nCheck your phone - you should see:")
    print("  - ONE notification that was updated 4 times")
    print("  - NOT 4 separate notifications")
    print("=" * 60)


if __name__ == "__main__":
    test_progress_updates()
