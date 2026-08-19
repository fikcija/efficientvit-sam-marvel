#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 07_extract_marvel_frames.sh — dekodovanje MARVEL videa u JPEG, jednom.
#
# Zašto dekodovati unapred umesto da se radi u hodu:
# MARVEL se šalje kroz pet varijanti modela plus perturbacione runove.
# Ponovno dekodovanje 22,985 H.264 frejmova na svakom prolazu troši CPU za byte-identičan
# output. Dekodovanje jednom u JPEG takođe čini da MARVEL pipeline strukturalno 
# identičan COCO pipeline-u: oba dataset-a se sada ponašaju isto — "read an image, prompt with a box".
#
#
# Output nazivi se poklapaju sa anotacijama: frame_000000.jpg se uparuje sa
# frame_000000.txt, pa se ne mora vršiti promena indeksa nigde downstream.
#
# Otprilike 2-4 GB. Pokrenuti jednom na podu, network volume čuva.
#
# Korišćenje:  bash /workspace/scripts/07_extract_marvel_frames.sh
# ---------------------------------------------------------------------------

set -uo pipefail

MARVEL="${MARVEL_ROOT:-/workspace/marvel/MULTIMODAL_DATASET}"
VIDEOS="${MARVEL}/anonymised_videos"
FRAMES="${MARVEL}/frames"

command -v ffmpeg >/dev/null || {
    echo "ffmpeg not found. Install it:  apt-get update && apt-get install -y ffmpeg"
    exit 1
}

[ -d "${VIDEOS}" ] || {
    echo "No videos at ${VIDEOS}"
    echo "Extract anonymised_videos.zip first."
    exit 1
}

mkdir -p "${FRAMES}"

total=0
for video in "${VIDEOS}"/*.mp4; do
    [ -e "${video}" ] || continue
    base="$(basename "${video}" .mp4)"
    out="${FRAMES}/${base}"

    if [ -d "${out}" ] && [ -n "$(ls -A "${out}" 2>/dev/null)" ]; then
        n=$(ls "${out}" | wc -l)
        echo "  ${base}: ${n} frames already present, skipping"
        total=$((total + n))
        continue
    fi

    mkdir -p "${out}"
    echo "  ${base}: decoding ..."

    # -start_number 0  time je prvi frejm: frame_000000.jpg
    # -q:v 2           JPEG visokog kvaliteta; ovde kvalitet nije usko grlo
    ffmpeg -nostdin -loglevel error -i "${video}" \
           -start_number 0 -q:v 2 \
           "${out}/frame_%06d.jpg"

    n=$(ls "${out}" | wc -l)
    echo "  ${base}: ${n} frames"
    total=$((total + n))
done

echo
echo "Total extracted: ${total} frames"
echo "Annotations expect: 22985"
echo
echo "A small mismatch is normal because some frames are dropped in the anonymisation process."
echo
du -sh "${FRAMES}" 2>/dev/null
