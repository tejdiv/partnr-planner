# Habitat-LLM Installation Instructions

### Requirements:
- Conda or Mamba
sudo apt-get update && sudo apt-get install -y git wget unzip build-essential

# 2. Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 3. Install Miniconda (Press Enter/Yes when prompted)
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

# 4. Initialize Conda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"
conda init

 conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


# 5. Reload shell to apply changes
source ~/.bashrc


git clone https://github.com/tejdiv/partnr-planner.git
cd partnr-planner


### Create and activate Conda environment
```bash
conda create -n habitat-llm  python=3.9.2 cmake=3.14.0 -y
conda activate habitat-llm
```

### Initialize third party submodules
```bash
git submodule sync
git submodule update --init --recursive
```

### Install dependencies and requirements
```bash
# Adjust the cuda version depending on your hardware stack
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Fix MKL compatibility issue (PyTorch requires MKL < 2024.1)
# This prevents "undefined symbol: iJIT_NotifyEvent" error
conda install "mkl<2024.1" "intel-openmp<2024.1" -y

# Verify PyTorch works
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# Install habitat-sim version 0.3.3
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y
# NOTE: If the above fails, packages may not be available for your system. Install from source (see https://github.com/facebookresearch/habitat-sim).
pip install -e ./third_party/habitat-lab/habitat-lab
pip install -e ./third_party/habitat-lab/habitat-baselines
pip install -e ./third_party/transformers-CFG
pip install -r requirements.txt
```
If you have issues with library linking make sure that the conda libraries are in your LD_LIBRARY_PATH (e.g `export LD_LIBRARY_PATH=/path/to/anaconda/envs/myenv/lib:$LD_LIBRARY_PATH`)

### Install NVIDIA OpenGL/EGL libraries (Lambda Labs and headless GPU servers only)
```bash
# Required for running Habitat-Sim on headless GPU servers like Lambda Labs
# Check your driver version first with: nvidia-smi
# Then install the matching OpenGL libraries (replace 570 with your driver version)
sudo apt-get update
sudo apt-get install -y libnvidia-gl-570-server libegl1 libgl1-mesa-glx libopengl0 git-lfs

git lfs install
# Reboot to sync driver and library versions
sudo reboot
```

After reboot, reconnect and reactivate your conda environment before continuing.

### Download datasets
```bash
# You may have to re-run downloader commands in case of network errors.
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets hab_spot_arm hab3-episodes habitat_humanoids --data-path data/ --no-replace --no-prune

# Download ovmm objects
git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects data/objects_ovmm --recursive
```

### Setup HSSD scene dataset
```bash
# Download and link the data.
git clone -b partnr https://huggingface.co/datasets/hssd/hssd-hab data/versioned_data/hssd-hab
cd data/versioned_data/hssd-hab
git lfs pull
cd ../../..
ln -s versioned_data/hssd-hab data/hssd-hab
```

### Download task datasets and neural network skill checkpoints

```bash
# Download the data
git clone https://huggingface.co/datasets/ai-habitat/partnr_episodes data/versioned_data/partnr_episodes
cd data/versioned_data/partnr_episodes
git lfs pull
cd ../../..

# Link task datasets
mkdir -p data/datasets
ln -s ../versioned_data/partnr_episodes data/datasets/partnr_episodes

# Link skill checkpoints
ln -s versioned_data/partnr_episodes/checkpoints data/models
```

### (Optional) Install pybullet for IK based controllers
```bash
pip install pybullet==3.0.4
```

### Install pre-commit
```bash
pip install pre-commit && pre-commit install
```

### Install the habitat-llm library
```bash
pip install -e .
```

### Setup api keys if needed:
```bash
# Add the following to your ~/.bashrc file
export OPENAI_API_KEY=...
```

### Running the HITL server (Lambda Labs and headless GPU servers)
```bash
# Set the EGL vendor library path (required on headless GPU servers)
export __EGL_VENDOR_LIBRARY_FILENAMES=~/partnr-planner/egl_vendor/10_nvidia.json
export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"

# Run the server
python habitat_llm/server/hitl_server.py
```

### Run the tests
```bash
# make sure to use bash shell if on zsh
bash

# Download and link the data.
git clone https://huggingface.co/datasets/ai-habitat/hssd-partnr-ci data/versioned_data/hssd-partnr-ci
ln -s versioned_data/hssd-partnr-ci data/hssd-partnr-ci
cd data/hssd-partnr-ci
git lfs pull
cd ../..

# link RAG testing data
ln -s versioned_data/partnr_episodes/test_rag data/test_rag

# then, run the tests
python -m pytest habitat_llm/tests [-v]
```

#### Troubleshooting

If the tests mentioned above fail due to scipy, numpy, pandas or opencv incompatibility, try installing the following alternate versions of selected packages which worked for us on Ubuntu 22.04.5 LTS:
```
pip install scipy==1.12.0 # make sure to run this before numpy or pandas install
pip install numpy==1.22.0
pip install pandas==2.0.3
pip install opencv-python==4.10.0.82
```
