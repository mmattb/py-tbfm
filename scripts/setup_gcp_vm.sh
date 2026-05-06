#!/bin/bash
# GCP VM Setup Script for py-tbfm project
# This script sets up a GCP VM for running the TBFM Test-Time Adaptation evaluation

set -e  # Exit on any error

echo "========================================="
echo "Setting up GCP VM for py-tbfm project"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Update system packages
print_status "Step 1: Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Step 2: Install basic development tools
print_status "Step 2: Installing basic development tools..."
sudo apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    unzip

# Step 3: Install Python 3.10 and development tools
print_status "Step 3: Installing Python 3.10 and development tools..."
sudo apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip

# Update pip
python3.10 -m pip install --upgrade pip

# Step 4: Check for GPU and CUDA
print_status "Step 4: Checking for GPU..."
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    print_warning "No NVIDIA GPU detected. Will install CUDA drivers."
    print_status "Installing NVIDIA drivers and CUDA toolkit..."

    # Add NVIDIA package repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update

    # Install CUDA toolkit (version 12.x for PyTorch 2.8.0)
    sudo apt-get install -y cuda-toolkit-12-8

    # Install NVIDIA drivers
    sudo apt-get install -y nvidia-driver-560

    print_warning "NVIDIA drivers installed. You may need to REBOOT the VM."
    print_warning "After reboot, run 'nvidia-smi' to verify the installation."
fi

# Step 5: Create Python virtual environment
print_status "Step 5: Creating Python virtual environment..."
cd /home/danmuir/GitHub/py-tbfm

if [ ! -d ".venv" ]; then
    python3.10 -m venv .venv
    print_status "Virtual environment created at .venv"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate

# Step 6: Install Python dependencies
print_status "Step 6: Installing Python dependencies..."
print_warning "This may take 10-20 minutes depending on your VM specs..."

# First install PyTorch with CUDA support
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies from requirements.txt
pip install -r requirements.txt

# Install the tbfm package in editable mode
pip install -e .

# Step 7: Setup data directory
print_status "Step 7: Setting up data directory..."
sudo mkdir -p /var/data/opto-coproc
sudo chown -R $USER:$USER /var/data
print_status "Data directory created at /var/data/opto-coproc"
print_warning "Remember to transfer your data to /var/data/opto-coproc/"

# Step 8: Create output directories
print_status "Step 8: Creating output directories..."
mkdir -p data/tta_sweep
mkdir -p test
print_status "Output directories created"

# Step 9: Verify CUDA installation
print_status "Step 9: Verifying PyTorch CUDA setup..."
python3.10 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}')"

# Step 10: Setup environment variables
print_status "Step 10: Setting up environment variables..."
cat >> ~/.bashrc << 'EOF'

# py-tbfm project environment
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Activate py-tbfm virtual environment
alias activate-tbfm='source /home/danmuir/GitHub/py-tbfm/.venv/bin/activate'
EOF

source ~/.bashrc

# Step 11: Create quick reference file
print_status "Step 11: Creating quick reference guide..."
cat > /home/danmuir/GitHub/py-tbfm/GCP_VM_REFERENCE.md << 'EOF'
# GCP VM Quick Reference for py-tbfm

## Activate Virtual Environment
```bash
cd /home/danmuir/GitHub/py-tbfm
source .venv/bin/activate
# Or use the alias:
activate-tbfm
```

## Check GPU Status
```bash
nvidia-smi
# Or for detailed info:
nvidia-smi -l 1  # Update every second
```

## Run TTA Testing
```bash
cd /home/danmuir/GitHub/py-tbfm
source .venv/bin/activate
python tta_testing.py --cuda-device 0 --help
```

## Example Run Commands
```bash
# Run with default settings
python tta_testing.py --cuda-device 0

# Run with specific support sizes
python tta_testing.py --cuda-device 0 --support-sizes 100 250 500

# Run with specific session
python tta_testing.py --cuda-device 0 --adapt-session "MonkeyG_20150914_Session1_S1"
```

## Monitor GPU Usage During Training
```bash
watch -n 1 nvidia-smi
```

## Check Python Package Versions
```bash
pip list | grep torch
pip list | grep cuda
```

## Data Directory
- Data location: `/var/data/opto-coproc/`
- Expected structure:
  - `/var/data/opto-coproc/{session_id}/torchraw/`
  - `/var/data/opto-coproc/{session_id}/embedding_rest/er.torch`

## Output Directory
- Results are saved to: `data/tta_sweep/`

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `--batch-size-per-session 5000`
- Use fewer sessions or smaller support sizes

### Missing Data Files
- Check data directory: `ls /var/data/opto-coproc/`
- Verify session structure matches expected format

### GPU Not Detected
- Check driver: `nvidia-smi`
- If no output, reboot: `sudo reboot`
- Reinstall drivers if needed

## Useful Commands
```bash
# Check disk space
df -h

# Check memory usage
free -h

# Monitor system resources
htop

# Check CUDA version
nvcc --version

# Check Python environment
which python
python --version
```
EOF

print_status "Quick reference created at GCP_VM_REFERENCE.md"

# Final summary
echo ""
echo "========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. If NVIDIA drivers were just installed, REBOOT the VM: sudo reboot"
echo "2. Transfer your data to: /var/data/opto-coproc/"
echo "3. Activate the environment: source .venv/bin/activate"
echo "4. Run a test: python tta_testing.py --help"
echo ""
echo "For more information, see: GCP_VM_REFERENCE.md"
echo ""
