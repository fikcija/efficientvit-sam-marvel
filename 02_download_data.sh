#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 02_download_data.sh — skinuti i raspakovati COCO i GRN MARVEL dataset-e, i preuzeti model checkpoint-e.
#
# Potrebno je pokrenuti samo jednom, jer sve ide unutar /workspace network volume-a, koji preživi gašenje pod-a. 
# Svaki korak se preskače ako je target već prisutan, ovim je skripta bezbedna za ponovni poziv nakon prekinutog skidanja.
#
# Total: otprilike 4-5 GB. Očekivano vreme trajanja 10-20 minuta.
#
# Lokacija:  bash /workspace/scripts/02_download_data.sh
# ---------------------------------------------------------------------------

set -euo pipefail

WORKSPACE="/workspace"
COCO_DIR="${WORKSPACE}/coco"
CKPT_DIR="${WORKSPACE}/efficientvit/assets/checkpoints/efficientvit_sam"
MARVEL_DIR="${WORKSPACE}/marvel"
HF_BASE="https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main"

mkdir -p "${COCO_DIR}/annotations" "${COCO_DIR}/source_json_file" "${CKPT_DIR}" "${MARVEL_DIR}"

# --- COCO val2017 slike ---------------------------------------------------
echo "=== COCO val2017 images (815 MB) ==="
if [ -d "${COCO_DIR}/val2017" ]; then
    echo "Already present ($(ls "${COCO_DIR}/val2017" | wc -l) files), skipping."
else
    wget -c -O /tmp/val2017.zip http://images.cocodataset.org/zips/val2017.zip
    unzip -q /tmp/val2017.zip -d "${COCO_DIR}"
    rm /tmp/val2017.zip
fi

# --- COCO anotacije ------------------------------------------------------
echo
echo "=== COCO annotations (241 MB) ==="
if [ -f "${COCO_DIR}/annotations/instances_val2017.json" ]; then
    echo "Already present, skipping."
else
    wget -c -O /tmp/ann.zip \
        http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q -j /tmp/ann.zip "annotations/instances_val2017.json" \
        -d "${COCO_DIR}/annotations"
    rm /tmp/ann.zip
fi

# --- Pre-generated detector boxes ------------------------------------------
# Reprodukcija tabela 4 i 5 bez instaliranja ili pokretanja bilo kojeg
# Objavljeno od strane autora rada.
echo
echo "=== Detector box files ==="
for f in coco_vitdet.json coco_yolov8.json coco_groundingdino.json; do
    if [ -f "${COCO_DIR}/source_json_file/${f}" ]; then
        echo "  ${f} already present, skipping."
    else
        echo "  downloading ${f}"
        wget -c -q --show-progress \
            -O "${COCO_DIR}/source_json_file/${f}" \
            "${HF_BASE}/source_json_file/${f}"
    fi
done

# --- Model checkpoints -----------------------------------------------------
echo
echo "=== EfficientViT-SAM checkpoints ==="
for m in l0 l1 l2 xl0 xl1; do
    target="${CKPT_DIR}/efficientvit_sam_${m}.pt"
    if [ -f "${target}" ]; then
        echo "  ${m} already present, skipping."
    else
        echo "  downloading ${m}"
        wget -c -q --show-progress -O "${target}" \
            "${HF_BASE}/efficientvit_sam_${m}.pt"
    fi
done

# --- GRN MARVEL ------------------------------------------------------------
# Ovo ship-uje arhive arhiva, koje su nestovane u tri nivoa:
#   marvel.zip
#     └── MULTIMODAL_DATASET/
#           ├── anonymised_videos.zip        19 .mp4
#           ├── bounding_box_annotation.zip  19 more .zip, one per clip
#           ├── extracted_audio.zip          not needed
#           └── audio_annotation.zip         not needed
# Sva tri nivoa moraju da se raspakuju pre nego što se bilo šta može pročitati.
echo
echo "=== GRN MARVEL dataset (461 MB) ==="
MM="${MARVEL_DIR}/MULTIMODAL_DATASET"

if [ -d "${MM}" ]; then
    echo "Outer archive already unpacked, skipping download."
else
    wget -c -O /tmp/marvel.zip \
        "https://zenodo.org/records/10671777/files/GRN_MARVEL_MULTIMODAL_DATASET.zip?download=1"
    unzip -q /tmp/marvel.zip -d "${MARVEL_DIR}"
    rm /tmp/marvel.zip
    wget -c -q -O "${MARVEL_DIR}/metadata.pdf" \
        "https://zenodo.org/records/10671777/files/12_GRN-AV-VRU_MARVEL_Metadata.pdf?download=1"
fi

echo "--- videos ---"
if [ -d "${MM}/anonymised_videos" ]; then
    echo "  already extracted ($(ls "${MM}/anonymised_videos" | wc -l) files)"
else
    unzip -q "${MM}/anonymised_videos.zip" -d "${MM}"
    echo "  $(ls "${MM}/anonymised_videos" | wc -l) videos"
fi

echo "--- bounding box annotations (nested one level deeper) ---"
BB="${MM}/bbox/bounding_box_annotation"
if [ -d "${BB}/extracted" ]; then
    echo "  already extracted ($(ls "${BB}/extracted" | wc -l) clips)"
else
    unzip -q "${MM}/bounding_box_annotation.zip" -d "${MM}/bbox"
    mkdir -p "${BB}/extracted"
    # one zip per clip; names contain spaces and parentheses, hence the quoting
    for z in "${BB}"/*.zip; do
        [ -e "${z}" ] || continue
        name="$(basename "${z}" .zip)"
        unzip -q -o "${z}" -d "${BB}/extracted/${name}"
    done
    echo "  $(ls "${BB}/extracted" | wc -l) clips"
fi

echo "  annotated frames: $(find "${BB}/extracted" -name 'frame_*.txt' | wc -l)  (expected 22985)"

# extracted_audio.zip i audio_annotation.zip ostaju zapakovani — jer su za projekat potrebni samo videi i bounding box anotacije.

# --- Report ----------------------------------------------------------------
echo
echo "=== Done. Contents of ${WORKSPACE} ==="
du -sh "${COCO_DIR}"/* "${CKPT_DIR}" "${MARVEL_DIR}" 2>/dev/null || true
echo
echo "COCO val images : $(ls "${COCO_DIR}/val2017" 2>/dev/null | wc -l)   (expected 5000)"
echo
echo "Next: bash ${WORKSPACE}/scripts/03_run_experiments.sh validate"
