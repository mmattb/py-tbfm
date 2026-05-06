# Live Notifications Quick Start

## Setup (2 minutes)

1. **Install dependencies**:
   ```bash
   pip install requests python-dotenv
   ```

2. **Get Credentials**:
   - Go to https://pushover.net and sign up ($5 one-time)
   - Install Pushover app on your iPhone
   - Copy your **User Key** from the dashboard
   - Click "Create Application", name it, copy the **API Token**

3. **Configure**:
   ```bash
   export NOTIFY_METHOD="pushover"
   export PUSHOVER_USER_KEY="your_user_key"
   export PUSHOVER_API_TOKEN="your_api_token"
   ```

4. **Test**:
   ```bash
   python test_notifications.py
   ```

## What You Get

### Training (tma_standalone.py)
- ✅ Start notification when training begins
- 📊 Progress updates every 1000 epochs (silent)
- ✅ Completion with final R² scores

### TTA (tta_testing.py)  
- ✅ Start notification for each run
- 📊 Progress updates every 500 steps (silent)
- ✅ Completion with R² for each model/strategy/support size

### Errors
- ⚠️ Instant HIGH PRIORITY notification if anything crashes

## Example Notifications

**Training Progress:**
```
Training 100_25_rr16
Epoch 3000/7001
Train Loss: 0.0234 | Test Loss: 0.0256
Train R²: 0.8523 | Test R²: 0.8234
```

**TTA Progress:**
```
TTA: 100_25_inner_ts5000
TTA Progress: 1500/7001
Loss: 0.0189
Mode: AE
```

**Completion:**
```
✓ Training Complete
Train R²: 0.8523
Test R²: 0.8234
Output: test/100_25_rr16_inner_ts5000
```

## Make it Permanent

Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export NOTIFY_METHOD="pushover"
export PUSHOVER_USER_KEY="your_user_key"
export PUSHOVER_API_TOKEN="your_api_token"
```

Then run: `source ~/.bashrc` (or `~/.zshrc`)

Now all training runs will send you live updates automatically! 🎉

## Why Pushover?

- ✅ **Works on iPhone** (Pushbullet dropped iOS support)
- ✅ Priority levels (errors = high, progress = silent)
- ✅ Desktop apps for Mac/Windows/Linux
- ✅ One-time $5 cost (no subscription)
- ✅ Super reliable
