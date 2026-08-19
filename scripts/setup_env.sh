#!/usr/bin/env bash
set -e

echo "=================================================="
echo "[setup] HumanoidVerse Genesis Demo Environment"
echo "=================================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-hgen}"

echo "[setup] project dir: $PROJECT_DIR"
echo "[setup] conda env  : $ENV_NAME"

echo
echo "[setup] installing Ubuntu packages for Genesis viewer..."
sudo apt update
sudo apt install -y \
  git \
  build-essential \
  mesa-utils \
  x11-apps \
  libopengl0 \
  libgl1 \
  libglx-mesa0 \
  libegl1 \
  libgl1-mesa-dri \
  libglu1-mesa

echo
echo "[setup] checking conda..."

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] conda.sh not found."
    echo "Please install Miniconda or Anaconda in WSL first."
    exit 1
fi

echo
echo "[setup] creating conda env if needed..."

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[setup] conda env '$ENV_NAME' already exists"
else
    conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"

echo
echo "[setup] upgrading pip tools..."
python -m pip install --upgrade pip setuptools wheel

echo
echo "[setup] installing PyTorch CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo
echo "[setup] installing local HumanoidVerse project..."
cd "$PROJECT_DIR"
pip install -e .

echo
echo "[setup] installing Genesis 0.2.1..."
pip uninstall -y genesis-world || true
pip install genesis-world==0.2.1

echo
echo "[setup] pinning numpy..."
pip install numpy==1.26.4

echo
echo "[setup] installing rsl_rl..."
pip install git+https://github.com/leggedrobotics/rsl_rl.git

echo
echo "[setup] installing extra dependencies..."
pip install tensorboard pynput rich termcolor matplotlib tqdm gymnasium

echo
echo "[setup] fixing Genesis viewer compatibility..."
pip uninstall -y libigl pyglet PyOpenGL PyOpenGL_accelerate || true

pip install --no-cache-dir \
  "libigl==2.4.1" \
  "pyglet==1.5.27" \
  "PyOpenGL==3.1.6"

echo
echo "=================================================="
echo "[setup] verifying installation"
echo "=================================================="

python - <<'PY'
import sys
import torch

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

try:
    import genesis as gs
    print("genesis:", getattr(gs, "__version__", "unknown"))
except Exception as e:
    print("genesis import failed:", e)

try:
    import numpy as np
    print("numpy:", np.__version__)
except Exception as e:
    print("numpy import failed:", e)

try:
    import igl
    print("libigl: OK")
except Exception as e:
    print("libigl import failed:", e)

try:
    import pyglet
    print("pyglet:", pyglet.version)
except Exception as e:
    print("pyglet import failed:", e)

try:
    import OpenGL
    print("PyOpenGL: OK")
except Exception as e:
    print("PyOpenGL import failed:", e)
PY

echo
echo "=================================================="
echo "[setup] done"
echo "Next:"
echo "  ./scripts/run_demo.sh"
echo
echo "For isolated test:"
echo "  ENV_NAME=hgen_submit_test ./scripts/run_demo.sh"
echo "=================================================="
