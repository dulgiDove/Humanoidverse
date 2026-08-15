#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-hgen}"

echo "=================================================="
echo "[run] Humanoid Navigation Challenge"
echo "=================================================="
echo "[run] project dir: $PROJECT_DIR"
echo "[run] conda env  : $ENV_NAME"

cd "$PROJECT_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] conda.sh not found."
    echo "Please install Miniconda or Anaconda in WSL first."
    exit 1
fi

conda activate "$ENV_NAME"

export PYTHONPATH="$PROJECT_DIR"
export PYOPENGL_PLATFORM=glx
export LIBGL_ALWAYS_INDIRECT=0
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA

case ":${LD_LIBRARY_PATH:-}:" in
    *:/usr/lib/wsl/lib:*) ;;
    *) export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" ;;
esac

unset LIBGL_ALWAYS_SOFTWARE
unset MESA_GL_VERSION_OVERRIDE
unset MESA_GLSL_VERSION_OVERRIDE

CKPT="${CKPT:-$(find logs/H1_E2E -name "model_*.pt" | sort -V | tail -1)}"

if [ -z "$CKPT" ]; then
    echo "[ERROR] checkpoint not found."
    echo "Expected: logs/H1_E2E/.../model_*.pt"
    exit 1
fi

if [ ! -f "$CKPT" ]; then
    echo "[ERROR] checkpoint file does not exist:"
    echo "$CKPT"
    exit 1
fi

echo "[run] checkpoint: $CKPT"
echo "[run] web panel : http://localhost:8080"
echo
echo "[run] Starting Genesis viewer and web control panel..."
echo "=================================================="

python humanoidverse/eval_web_control.py \
  +simulator=genesis \
  +exp=locomotion \
  +domain_rand=NO_domain_rand \
  +rewards=loco/reward_h1_locomotion \
  +robot=h1/h1_10dof \
  +terrain=terrain_locomotion_plane \
  +obs=loco/leggedloco_obs_singlestep_withlinvel \
  +num_envs=1 \
  +headless=False \
  +checkpoint="$CKPT"
