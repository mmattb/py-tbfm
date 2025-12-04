#!/usr/bin/env python3
"""
Notification utility for training completion and errors.

Supports Pushover push notifications with live progress updates.

Configure via environment variables:
    NOTIFY_METHOD: "pushover"
    PUSHOVER_USER_KEY: Your user key from https://pushover.net
    PUSHOVER_API_TOKEN: Your application API token

Setup:
    1. Sign up at https://pushover.net (one-time $5 per platform)
    2. Install Pushover app on iPhone/Android
    3. Get your User Key from the dashboard
    4. Create an application to get an API Token
    5. pip install python-dotenv requests
    6. Create .env file or export NOTIFY_METHOD="pushover"
    7. Set PUSHOVER_USER_KEY="your_user_key"
    8. Set PUSHOVER_API_TOKEN="your_api_token"
"""

import os
import traceback
from typing import Optional

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables


class Notifier:
    """Handles sending notifications via multiple methods."""
    
    def __init__(self):
        self.method = os.getenv("NOTIFY_METHOD", "").lower()
        self.enabled = bool(self.method)
        
    def send(self, subject: str, message: str, is_error: bool = False):
        """
        Send a notification.
        
        Args:
            subject: Notification subject/title
            message: Notification message body
            is_error: Whether this is an error notification
        """
        if not self.enabled:
            print(f"[Notification] {subject}: {message}")
            return
            
        try:
            if self.method == "pushover":
                priority = 1 if is_error else 0  # High priority for errors
                self._send_pushover(subject, message, priority=priority)
            else:
                print(f"[Notification] Unknown method: {self.method}")
                print(f"[Notification] {subject}: {message}")
        except Exception as e:
            print(f"[Notification] Failed to send: {e}")
            print(f"[Notification] {subject}: {message}")
    
    def _send_pushover(self, title: str, message: str, priority: int = 0):
        """Send notification via Pushover using HTTP API.
        
        Args:
            title: Notification title
            message: Notification message
            priority: Priority level (-2=silent, -1=quiet, 0=normal, 1=high, 2=emergency)
        """
        try:
            import requests
        except ImportError:
            raise ImportError("Install requests: pip install requests")
        
        user_key = os.getenv("PUSHOVER_USER_KEY")
        api_token = os.getenv("PUSHOVER_API_TOKEN")
        
        if not all([user_key, api_token]):
            raise ValueError("Missing PUSHOVER_USER_KEY or PUSHOVER_API_TOKEN")
        
        url = "https://api.pushover.net/1/messages.json"
        data = {
            "token": api_token,
            "user": user_key,
            "title": title,
            "message": message,
            "priority": priority,
        }
        
        response = requests.post(url, data=data)
        response.raise_for_status()
        
        priority_str = {-2: "silent", -1: "quiet", 0: "normal", 1: "high", 2: "emergency"}.get(priority, str(priority))
        print(f"[Notification] Pushover notification sent (priority: {priority_str})")


# Global notifier instance
_notifier = Notifier()


def notify_training_complete(model_name: str, metrics: dict, output_dir: str = None):
    """Notify when training completes successfully."""
    message_lines = [f"Model: {model_name}"]
    
    if metrics:
        if "train_r2" in metrics:
            message_lines.append(f"Train R²: {metrics['train_r2']:.4f}")
        if "test_r2" in metrics:
            message_lines.append(f"Test R²: {metrics['test_r2']:.4f}")
        if "final_loss" in metrics:
            message_lines.append(f"Final Loss: {metrics['final_loss']:.4f}")
    
    if output_dir:
        message_lines.append(f"Output: {output_dir}")
    
    message = "\n".join(message_lines)
    _notifier.send("✓ Training Complete", message, is_error=False)


def notify_tta_complete(model_name: str, session: str, support_size: int, 
                       strategy: str, r2: float, output_dir: str = None):
    """Notify when TTA completes successfully."""
    message_lines = [
        f"Model: {model_name}",
        f"Session: {session}",
        f"Strategy: {strategy}",
        f"Support Size: {support_size}",
        f"Test R²: {r2:.4f}"
    ]
    
    if output_dir:
        message_lines.append(f"Output: {output_dir}")
    
    message = "\n".join(message_lines)
    _notifier.send("✓ TTA Complete", message, is_error=False)


def notify_error(script_name: str, error: Exception, context: str = None):
    """Notify when an error occurs."""
    message_lines = [f"Script: {script_name}"]
    
    if context:
        message_lines.append(f"Context: {context}")
    
    message_lines.append(f"\nError: {type(error).__name__}: {str(error)}")
    
    # Add traceback
    tb = traceback.format_exc()
    if tb and tb != "NoneType: None\n":
        message_lines.append(f"\nTraceback:\n{tb[:500]}")  # Limit traceback length
    
    message = "\n".join(message_lines)
    _notifier.send("✗ Training Error", message, is_error=True)


def is_enabled():
    """Check if notifications are enabled."""
    return _notifier.enabled


def create_progress_notification(job_id: str, title: str, initial_message: str):
    """
    Create a notification for progress tracking.
    
    Note: Pushover doesn't support updating existing notifications.
    Updates will arrive as separate (but silent) notifications.
    
    Args:
        job_id: Unique identifier for this job (for tracking)
        title: Notification title
        initial_message: Initial message content
        
    Returns:
        job_id if successful, None otherwise
    """
    if not _notifier.enabled:
        print(f"[Progress] {title}: {initial_message}")
        return None
    
    try:
        if _notifier.method == "pushover":
            # Send initial notification with normal priority (makes sound)
            _notifier._send_pushover(title, initial_message, priority=0)
            return job_id
        else:
            # For other methods, just send a notification
            _notifier.send(title, initial_message)
            return None
    except Exception as e:
        print(f"[Progress] Failed to create progress notification: {e}")
        print(f"[Progress] {title}: {initial_message}")
        return None


def update_progress_notification(job_id: str, message: str, title: str = None):
    """
    Send a progress update notification.
    
    Note: This sends a new silent notification (no sound/vibration).
    Pushover doesn't support updating existing notifications.
    
    Args:
        job_id: Job ID returned from create_progress_notification
        message: Updated message content
        title: Optional new title
    """
    if not job_id or not _notifier.enabled:
        print(f"[Progress Update] {message}")
        return
    
    try:
        if _notifier.method == "pushover":
            # Send quiet update (priority -1 = quiet, no sound but does vibrate/show)
            notification_title = title or "Progress Update"
            _notifier._send_pushover(notification_title, message, priority=-1)
            print(f"[Progress] Sent quiet update: {message[:50]}...")
        else:
            print(f"[Progress Update] {message}")
    except Exception as e:
        print(f"[Progress] Failed to update notification: {e}")
        print(f"[Progress Update] {message}")


def complete_progress_notification(job_id: str, final_message: str, title: str = None):
    """
    Mark a progress notification as complete and clean up.
    
    Sends a final update with normal priority and then removes the tag
    so future notifications don't replace this completion message.
    
    Args:
        job_id: Job ID returned from create_progress_notification
        final_message: Final completion message
        title: Optional final title
    """
    if not job_id or not _notifier.enabled:
        print(f"[Progress Complete] {final_message}")
        return
    
    try:
        if _notifier.method == "pushover":
            final_title = title or "✓ Complete"
            # Send final notification with normal priority (makes sound)
            _notifier._send_pushover(final_title, final_message, priority=0)
        else:
            print(f"[Progress Complete] {final_message}")
    except Exception as e:
        print(f"[Progress] Failed to complete notification: {e}")
        print(f"[Progress Complete] {final_message}")
