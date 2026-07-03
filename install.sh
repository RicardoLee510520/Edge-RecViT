
set -Eeuo pipefail

ENV_NAME="DASH_Vision_Transformer"
PY_VER="3.10"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

log() { echo "[install] $(date '+%F %T') $*"; }


if command -v module &>/dev/null; then
  if module avail 2>&1 | grep -qi "anaconda"; then
    log "Loading anaconda3 module..."
    module load anaconda3 || true
  fi
fi


if command -v conda &>/dev/null; then
  log "Conda detected. Creating env: ${ENV_NAME} (python=${PY_VER})"
  eval "$(conda shell.bash hook)"
  if conda env list | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
    log "Conda env ${ENV_NAME} already exists. Skipping creation."
  else
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
  fi
  conda activate "${ENV_NAME}"
else
  log "Conda not found. Using venv..."
  python3 -m venv morvit_env
  # shellcheck disable=SC1091
  source morvit_env/bin/activate
fi

python -m pip install -U pip wheel setuptools --no-cache-dir

# 安装 PyTorch + CUDA 12.1 固定版本
log "Installing PyTorch 2.5.1 + cu121 and torchvision 0.20.1..."
pip install --no-cache-dir \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

if [[ -f "requirements.txt" ]]; then
  log "Installing project requirements from requirements.txt..."
  pip install --no-cache-dir -r requirements.txt
else
  log "requirements.txt not found. Skipping."
fi


log "Pinning base dependencies..."
pip install --no-cache-dir \
  numpy==1.26.4 \
  Pillow==11.1.0

log "Environment setup complete!"
log "To activate: conda activate ${ENV_NAME}  (or source morvit_env/bin/activate)"
