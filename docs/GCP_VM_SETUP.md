# GCP VM Setup Guide for py-tbfm

This guide will help you set up a Google Cloud Platform (GCP) VM for running the py-tbfm Test-Time Adaptation evaluation project.

## Prerequisites

### 1. GCP VM Requirements
- **Instance Type**: GPU-enabled instance (e.g., `n1-standard-8` with T4/V100/A100 GPU)
- **OS**: Ubuntu 22.04 LTS
- **Disk**: At least 100GB boot disk + additional storage for data
- **Region**: Choose a region with GPU availability

### 2. Recommended GCP Configuration
```bash
# Create a VM with GPU using gcloud CLI (example)
gcloud compute instances create tbfm-gpu-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE
```

## Quick Setup (Automated)

### Option 1: Run the Setup Script
The easiest way to set up your VM is to use the provided setup script:

```bash
# SSH into your GCP VM
gcloud compute ssh tbfm-gpu-vm --zone=us-central1-a

# Clone the repository (if not already done)
cd ~
mkdir -p GitHub
cd GitHub
git clone https://github.com/mmattb/py-tbfm.git
cd py-tbfm

# Run the setup script
./setup_gcp_vm.sh
```

The script will:
1. Update system packages
2. Install Python 3.10 and development tools
3. Install CUDA drivers and toolkit (if not present)
4. Create a Python virtual environment
5. Install all Python dependencies
6. Set up data directories
7. Configure environment variables

**IMPORTANT**: If the script installs NVIDIA drivers, you'll need to reboot:
```bash
sudo reboot
```

After reboot, verify GPU is working:
```bash
nvidia-smi
```

## Manual Setup

If you prefer to set up manually or need to troubleshoot, follow these steps:

### Step 1: Update System
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Step 2: Install CUDA Drivers
```bash
# Check if GPU is detected
lspci | grep -i nvidia

# Install NVIDIA drivers
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8 nvidia-driver-560

# Reboot
sudo reboot

# Verify installation
nvidia-smi
```

### Step 3: Install Python Environment
```bash
cd /home/danmuir/GitHub/py-tbfm

# Install Python 3.10
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA support
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Step 4: Verify CUDA in PyTorch
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

### Step 5: Set Up Data Directory
```bash
sudo mkdir -p /var/data/opto-coproc
sudo chown -R $USER:$USER /var/data
```

## Data Transfer

Transfer your data to the VM using one of these methods:

### Option 1: Using gsutil (recommended for GCS)
```bash
# Install gcloud SDK if not present
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Copy data from Google Cloud Storage
gsutil -m cp -r gs://your-bucket/opto-coproc/* /var/data/opto-coproc/
```

### Option 2: Using gcloud compute scp
```bash
# From your local machine
gcloud compute scp --recurse /path/to/opto-coproc/* \
    tbfm-gpu-vm:/var/data/opto-coproc/ \
    --zone=us-central1-a
```

### Option 3: Using rsync over SSH
```bash
# From your local machine
gcloud compute config-ssh
rsync -avz --progress /path/to/opto-coproc/ \
    tbfm-gpu-vm.us-central1-a:/var/data/opto-coproc/
```

## Expected Data Structure
```
/var/data/opto-coproc/
├── {session_id_1}/
│   ├── torchraw/
│   └── embedding_rest/
│       └── er.torch
├── {session_id_2}/
│   ├── torchraw/
│   └── embedding_rest/
│       └── er.torch
└── ...
```

## Running the Tests

### Activate Environment
```bash
cd /home/danmuir/GitHub/py-tbfm
source .venv/bin/activate
```

### Basic Test Run
```bash
# Run with default settings
python tta_testing.py --cuda-device 0

# Run with specific parameters
python tta_testing.py \
    --cuda-device 0 \
    --support-sizes 100 250 500 1000 \
    --tta-epochs 7001 \
    --batch-size-per-session 7500
```

### Monitor GPU During Execution
```bash
# In a separate terminal/tmux pane
watch -n 1 nvidia-smi
```

## Troubleshooting

### GPU Not Detected
```bash
# Check if GPU is visible
lspci | grep -i nvidia

# Check driver status
nvidia-smi

# If driver not loaded
sudo modprobe nvidia

# Reinstall drivers if needed
sudo apt-get purge nvidia-*
sudo apt-get install -y nvidia-driver-560
sudo reboot
```

### CUDA Out of Memory
- Reduce `--batch-size-per-session` (default: 7500, try 5000 or 3000)
- Process fewer sessions at once
- Use smaller support sizes
- Monitor memory: `nvidia-smi -l 1`

### Missing Dependencies
```bash
# Reinstall all dependencies
source .venv/bin/activate
pip install --upgrade -r requirements.txt
```

### Data Not Found
```bash
# Check data directory structure
ls -la /var/data/opto-coproc/

# Verify permissions
sudo chown -R $USER:$USER /var/data
```

## Performance Optimization

### 1. Use Preemptible/Spot Instances
Save costs by using preemptible instances for non-critical runs:
```bash
gcloud compute instances create tbfm-gpu-vm \
    --preemptible \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1
```

### 2. Use Persistent Disk Snapshots
Create snapshots of your configured VM to quickly spin up new instances:
```bash
# Create snapshot of boot disk
gcloud compute disks snapshot tbfm-gpu-vm \
    --snapshot-names=tbfm-configured \
    --zone=us-central1-a
```

### 3. Use tmux for Long-Running Jobs
```bash
# Install tmux
sudo apt-get install tmux

# Start a session
tmux new -s tbfm

# Run your training
python tta_testing.py --cuda-device 0

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t tbfm
```

### 4. Optimize I/O with Local SSD
For better I/O performance, consider attaching a local SSD:
```bash
gcloud compute instances create tbfm-gpu-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --local-ssd interface=nvme
```

## Cost Management

### Estimated Costs (as of 2025)
- **n1-standard-8**: ~$0.38/hour
- **T4 GPU**: ~$0.35/hour
- **Total**: ~$0.73/hour (~$17.50/day for 24-hour run)

### Tips to Reduce Costs
1. Stop (don't delete) instances when not in use
2. Use preemptible instances for development
3. Monitor costs in GCP Console
4. Delete unused persistent disks
5. Use committed use discounts for long-term projects

## Additional Resources

- [GCP GPU Documentation](https://cloud.google.com/compute/docs/gpus)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

## Support

For issues specific to this project, check:
- Project README: [README.md](README.md)
- Quick Reference: [GCP_VM_REFERENCE.md](GCP_VM_REFERENCE.md)
- GitHub Issues: https://github.com/mmattb/py-tbfm/issues