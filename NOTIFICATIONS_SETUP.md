# Notification Setup Guide

Your training scripts (`tma_standalone.py` and `tta_testing.py`) now support notifications when training completes or errors occur.

## Setup: Pushover (iOS & Android)

Pushover is perfect for server notifications with excellent iOS support!

**Cost**: One-time $5 per platform (iOS or Android)

1. **Sign up**: Go to https://pushover.net
2. **Install app**: Download Pushover from App Store (iOS) or Google Play (Android)
3. **Get User Key**: After signing up, your User Key is shown on the dashboard
4. **Create Application**:
   - Click "Create an Application/API Token"
   - Name it (e.g., "TBFM Training")
   - Copy the API Token
5. **Install Python libraries**:
```bash
pip install requests python-dotenv
```

6. **Set environment variables** (choose one method):
```bash
export NOTIFY_METHOD="pushover"
export PUSHOVER_USER_KEY="your_user_key_from_dashboard"
export PUSHOVER_API_TOKEN="your_app_token"
```

## Persisting Configuration

Add to your shell profile so it's always available:

```bash
# For bash (add to ~/.bashrc or ~/.bash_profile)
echo 'export NOTIFY_METHOD="pushover"' >> ~/.bashrc
echo 'export PUSHOVER_USER_KEY="your_user_key"' >> ~/.bashrc
echo 'export PUSHOVER_API_TOKEN="your_api_token"' >> ~/.bashrc
source ~/.bashrc

# For zsh (add to ~/.zshrc)
echo 'export NOTIFY_METHOD="pushover"' >> ~/.zshrc
echo 'export PUSHOVER_USER_KEY="your_user_key"' >> ~/.zshrc
echo 'export PUSHOVER_API_TOKEN="your_api_token"' >> ~/.zshrc
source ~/.zshrc
```

Or create a `.env` file in your project directory:
```bash
# .env file
NOTIFY_METHOD=pushover
PUSHOVER_USER_KEY=your_user_key
PUSHOVER_API_TOKEN=your_api_token
```

Then load it before running:
```bash
source .env
python tma_standalone.py 100 25 0 false 16 5000 true
```

## Testing Notifications

Test your setup:

```bash
python test_notifications.py
```

This will send test notifications to verify everything is working.

## What You'll Get

### Live Training Updates (tma_standalone.py)
- **Start notification**: When training begins
- **Progress updates**: Every 1000 epochs with current train/test R² and loss
- **Completion notification**: Final R² scores and output directory

### Live TTA Updates (tta_testing.py)
- **Start notification**: When each TTA run begins (model + strategy + support size)
- **Progress updates**: Every 500 steps with current loss and optimization mode
- **Completion notification**: Final R² for each TTA run

### Error Notifications
- Immediate notification if any script crashes
- Includes error type, message, and traceback
- High priority so they stand out

All notifications appear on all your devices with Pushover installed!

## Priority Levels

- **Low priority** (-1): Progress updates (silent, no sound)
- **Normal priority** (0): Start and completion notifications
- **High priority** (1): Error notifications (bypasses quiet hours)

## Disabling Notifications

To disable notifications:

```bash
unset NOTIFY_METHOD
# or
export NOTIFY_METHOD=""
```

The scripts will still run normally and print messages to the console.

## Troubleshooting

### "Install requests" error
```bash
pip install requests
```

Note: We use the Pushover HTTP API directly (via requests) instead of the outdated python-pushover package.

### Invalid credentials
- Double-check you copied both User Key AND API Token
- User Key: from main dashboard at pushover.net
- API Token: from "Create Application" page
- Make sure there are no extra spaces

### Not receiving notifications
- Check that the Pushover app is installed on your device
- Try sending a test notification from the Pushover website
- Check your notification settings in the app
- Verify environment variables: 
  ```bash
  echo $PUSHOVER_USER_KEY
  echo $PUSHOVER_API_TOKEN
  ```

### Too many notifications
Progress updates are sent with low priority (silent). You can adjust frequency by:
- Training: Updates every 1000 epochs
- TTA: Updates every 500 steps

Edit `tbfm/multisession.py` to change the intervals if needed.

## Why Pushover?

- ✅ **iOS support** (Pushbullet dropped this)
- ✅ **Android support**
- ✅ **Desktop clients** (Windows, macOS, Linux)
- ✅ **Priority levels** (errors get high priority)
- ✅ **Reliable** (no rate limits for normal use)
- ✅ **One-time cost** ($5) instead of subscription
- ✅ **API is simple and fast**
