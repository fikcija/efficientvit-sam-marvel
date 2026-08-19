#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 01_setup_pod.sh — pokreće se pre svake sesije da se pripremi okruženje na GPU pod-u.
#
# Instalirani paketi žive na privremenom disku pod-a i nestaju kada se pod ugasi, tako da ovo mora da se pokrene za svaku sesiju. 
# Sam repozitorijum živi na network volume-u, tako da se kloniranje preskače ako je već tamo.
#
# Lokacija:  bash /workspace/scripts/01_setup_pod.sh
# ---------------------------------------------------------------------------

set -euo pipefail

WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/efficientvit"

echo "=== Checking we are on the network volume ==="
if [ ! -d "${WORKSPACE}" ]; then
    echo "ERROR: ${WORKSPACE} does not exist."
    echo "The network volume is not attached. Terminate this pod and redeploy"
    echo "with the volume selected."
    exit 1
fi
df -h "${WORKSPACE}"
echo

echo "=== GPU ==="
nvidia-smi || { echo "ERROR: no GPU visible"; exit 1; }
echo

echo "=== Cloning EfficientViT repo ==="
if [ -d "${REPO_DIR}/.git" ]; then
    echo "Repo already present at ${REPO_DIR}, skipping clone."
else
    git clone https://github.com/mit-han-lab/efficientvit.git "${REPO_DIR}"
fi
cd "${REPO_DIR}"
echo "Repo commit: $(git rev-parse --short HEAD)" #podsetnik: dodati u izveštaj
echo

echo "=== Installing dependencies ==="
pip install --quiet --upgrade pip
pip install -r requirements.txt
pip install --quiet pycocotools   # potrebno za COCO evaluaciju

echo
echo "=== Environment summary ===" #podsetnik: dodati u izveštaj
python - <<'PY'
import torch, platform
print("python      :", platform.python_version())
print("torch       :", torch.__version__)
print("cuda avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda version:", torch.version.cuda)
    print("gpu         :", torch.cuda.get_device_name(0))
    print("vram (GB)   :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
PY

echo
echo "Setup complete. Next: bash ${WORKSPACE}/scripts/02_download_data.sh"
