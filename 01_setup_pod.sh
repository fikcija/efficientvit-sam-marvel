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
# RunPod image dolazi sa torch-om build-ovanim za tačno taj CUDA driver. Ako
# repo pinuje drugu verziju, pip će je tiho zameniti i rezultat možda više
# neće videti GPU. Zato se torch/torchvision/torchaudio filtriraju iz
# requirements.txt i ostaje ono što image nudi.
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)

pip install --quiet --upgrade pip
grep -viE '^[[:space:]]*(torch|torchvision|torchaudio)([=<>!~[:space:]]|$)' \
    requirements.txt > /tmp/requirements_no_torch.txt
echo "  (preskačem torch/torchvision/torchaudio — koristi se verzija iz image-a)"
pip install -r /tmp/requirements_no_torch.txt
pip install --quiet pycocotools lvis   # potrebno za COCO i LVIS evaluaciju

TORCH_AFTER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ "${TORCH_BEFORE}" != "${TORCH_AFTER}" ]; then
    echo
    echo "  UPOZORENJE: torch je promenjen sa ${TORCH_BEFORE} na ${TORCH_AFTER}."
fi
python -c "import torch; assert torch.cuda.is_available()" \
    || { echo "GREŠKA: torch više ne vidi GPU."; exit 1; }

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
