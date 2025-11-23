# Fixing EGL Context Errors on Lambda Labs (Habitat-Sim)

## Problem
When running Habitat-Sim on Lambda Labs headless GPU instances, you may encounter:
```
Platform::WindowlessEglApplication::tryCreateContext(): unable to find CUDA device 0 among X EGL devices
WindowlessContext: Unable to create windowless context
```
or
```
Platform::WindowlessEglApplication::tryCreateContext(): cannot get default EGL display: EGL_BAD_PARAMETER
```

## Root Cause
Lambda Labs instances use `nvidia-headless-*-server` drivers optimized for CUDA compute workloads. These **do not include OpenGL/EGL libraries** required by Habitat-Sim for rendering.

## Solution

### Step 1: Install NVIDIA OpenGL/EGL Libraries
```bash
sudo apt-get update
sudo apt-get install -y libnvidia-gl-570-server
```

> **Note**: Replace `570` with your actual driver version. Check with `nvidia-smi` or `dpkg -l | grep nvidia-headless`

### Step 2: Create EGL Vendor Configuration File
```bash
# This file already exists in the repo at egl_vendor/10_nvidia.json
# If you need to create it manually:
mkdir -p ~/partnr-planner/egl_vendor
cat > ~/partnr-planner/egl_vendor/10_nvidia.json << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF
```

### Step 3: Set Environment Variables and Run
```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=~/partnr-planner/egl_vendor/10_nvidia.json
export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"
python habitat_llm/server/hitl_server.py
```

## Verification
After installing, verify the libraries are present:
```bash
ldconfig -p | grep -i egl | grep nvidia
```

You should see:
- `libEGL_nvidia.so.0`
- `libnvidia-eglcore.so.*`
- `libnvidia-egl-*.so.*`

## References
- [Habitat-Sim Issue #2424](https://github.com/facebookresearch/habitat-sim/issues/2424) - Missing EGL vendor config
- [Habitat-Sim Issue #1511](https://github.com/facebookresearch/habitat-sim/issues/1511) - EGL_BAD_PARAMETER on headless servers

## Quick Reference
For new Lambda Labs instances, run these commands:
```bash
# Install OpenGL libraries
sudo apt-get update && sudo apt-get install -y libnvidia-gl-570-server

# Set environment and run
export __EGL_VENDOR_LIBRARY_FILENAMES=~/partnr-planner/egl_vendor/10_nvidia.json
export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"
python habitat_llm/server/hitl_server.py
```