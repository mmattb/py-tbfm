#!/usr/bin/env python3
"""
Test notification system to verify setup is working correctly.
"""

import sys
import time
import notifications

def test_progress_updates():
    """Test that progress updates replace the same notification."""
    
    if not notifications.is_enabled():
        print("⚠️  Notifications are NOT enabled!")
        return False
    
    print("\n" + "=" * 60)
    print("Testing progress notification updates...")
    print("You should see ONE notification that updates 4 times, not 4 separate notifications.\n")
    
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
    
    print("\n✓ Progress update test complete!")
    print("Check your phone - you should see ONE notification that was updated 4 times")
    print("=" * 60)
    return True


def test_notifications():
    """Test all notification types."""
    
    if not notifications.is_enabled():
        print("⚠️  Notifications are NOT enabled!")
        print("\nTo enable notifications, set the NOTIFY_METHOD environment variable.")
        print("See NOTIFICATIONS_SETUP.md for detailed instructions.")
        print("\nExample:")
        print('  export NOTIFY_METHOD="email"')
        print('  export NOTIFY_EMAIL_TO="5551234567@txt.att.net"')
        print('  export SMTP_USER="your-email@gmail.com"')
        print('  export SMTP_PASSWORD="your-app-password"')
        sys.exit(1)
    
    print("✓ Notifications are enabled!")
    print(f"  Method: {notifications._notifier.method}")
    print("\nSending test notifications...\n")
    
    # Test 1: Training completion
    print("1. Testing training completion notification...")
    try:
        notifications.notify_training_complete(
            model_name="test_model_100_25",
            metrics={
                "train_r2": 0.8523,
                "test_r2": 0.8234
            },
            output_dir="test/100_25_rr16_inner_ts5000"
        )
        print("   ✓ Training completion notification sent!\n")
    except Exception as e:
        print(f"   ✗ Failed: {e}\n")
        return False
    
    # Test 2: TTA completion
    print("2. Testing TTA completion notification...")
    try:
        notifications.notify_tta_complete(
            model_name="test_model_100_25",
            session="MonkeyG_20150914_Session1_S1",
            support_size=1000,
            strategy="Co-Adaptation",
            r2=0.7845,
            output_dir="data/tta_sweep"
        )
        print("   ✓ TTA completion notification sent!\n")
    except Exception as e:
        print(f"   ✗ Failed: {e}\n")
        return False
    
    # Test 3: Error notification
    print("3. Testing error notification...")
    try:
        test_error = RuntimeError("This is a test error - everything is working!")
        notifications.notify_error(
            script_name="test_notifications.py",
            error=test_error,
            context="Testing notification system"
        )
        print("   ✓ Error notification sent!\n")
    except Exception as e:
        print(f"   ✗ Failed: {e}\n")
        return False
    
    print("=" * 60)
    print("✓ Basic notification tests passed!")
    print("\nYou should have received 3 notifications:")
    print("  1. Training completion")
    print("  2. TTA completion")
    print("  3. Error notification")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_notifications()
    
    if success:
        # Test progress updates
        test_progress_updates()
