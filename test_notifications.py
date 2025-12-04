#!/usr/bin/env python3
"""
Test notification system to verify setup is working correctly.
"""

import sys
import notifications

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
    print("✓ All tests passed!")
    print("\nYou should receive 3 notifications:")
    print("  1. Training completion")
    print("  2. TTA completion")
    print("  3. Error notification")
    print("\nIf you didn't receive them, check:")
    print("  - Your spam/junk folder (for email)")
    print("  - Environment variables are set correctly")
    print("  - See NOTIFICATIONS_SETUP.md for troubleshooting")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_notifications()
