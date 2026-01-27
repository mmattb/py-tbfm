# GCP VM Setup Checklist

Use this checklist to ensure your GCP VM is properly configured for py-tbfm.

## Pre-Setup

- [ ] GCP VM created with GPU (T4/V100/A100)
- [ ] Ubuntu 22.04 LTS installed
- [ ] At least 100GB disk space available
- [ ] SSH access configured

## Initial Setup

- [ ] Repository cloned to `/home/danmuir/GitHub/py-tbfm`
- [ ] Setup script is executable: `chmod +x setup_gcp_vm.sh`
- [ ] Setup script executed: `./setup_gcp_vm.sh`
- [ ] VM rebooted if NVIDIA drivers were installed
- [ ] GPU detected: `nvidia-smi` shows GPU info

## Environment Verification

- [ ] Python 3.10 installed: `python3.10 --version`
- [ ] Virtual environment created: `.venv/` directory exists
- [ ] Virtual environment activated: `source .venv/bin/activate`
- [ ] PyTorch installed: `python -c "import torch; print(torch.__version__)"`
- [ ] CUDA available in PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Package installed: `pip show tbfm` shows package info

## Data Setup

- [ ] Data directory created: `/var/data/opto-coproc/` exists
- [ ] Correct permissions: `ls -la /var/data/` shows your user as owner
- [ ] Data transferred to `/var/data/opto-coproc/`
- [ ] Data structure verified (session folders with `torchraw/` and `embedding_rest/`)

## Configuration

- [ ] Output directories created: `data/tta_sweep/`, `test/`
- [ ] Configuration files present: `conf/` directory exists
- [ ] Environment variables set in `~/.bashrc`

## Testing

- [ ] Help command works: `python tta_testing.py --help`
- [ ] Can list available devices: `python -c "import torch; print(torch.cuda.device_count())"`
- [ ] Quick test run (optional): `python tta_testing.py --cuda-device 0 --support-sizes 50 --tta-epochs 10`

## Optional Optimizations

- [ ] tmux installed for persistent sessions: `sudo apt-get install tmux`
- [ ] Monitoring tools installed: `htop`, `watch`
- [ ] GCP snapshot created for backup
- [ ] Cost monitoring set up in GCP Console

## Troubleshooting Reference

If any step fails, refer to:
- **Detailed guide**: [GCP_VM_SETUP.md](GCP_VM_SETUP.md)
- **Quick reference**: [GCP_VM_REFERENCE.md](GCP_VM_REFERENCE.md)
- **Setup script**: [setup_gcp_vm.sh](setup_gcp_vm.sh)

## Common Issues

### GPU Not Showing
```bash
nvidia-smi  # Should show GPU info
sudo reboot  # If no GPU shown
```

### Python Module Not Found
```bash
source .venv/bin/activate
pip install -e .
```

### CUDA Version Mismatch
```bash
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}'); print(f'System CUDA: ', end='')"; nvcc --version
```

### Permission Denied on Data Directory
```bash
sudo chown -R $USER:$USER /var/data
```

## Ready to Run!

Once all boxes are checked, you're ready to run experiments:

```bash
cd /home/danmuir/GitHub/py-tbfm
source .venv/bin/activate
python tta_testing.py --cuda-device 0
```

Monitor GPU usage in another terminal:
```bash
watch -n 1 nvidia-smi
```
