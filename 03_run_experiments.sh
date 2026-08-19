#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 03_run_experiments.sh — matrica eksperimenata za stavku 3.
#
# Svako pokretanje se upisuje u svoj vremenski označeni log pod /workspace/logs/, i
# pokretanje se preskace ako već postoji log. 
#
# Korišćenje:
#   bash 03_run_experiments.sh validate   # samo xl1 + GT boxes. Ovo ide prvo.
#   bash 03_run_experiments.sh table3     # svih pet varijanti, GT boxes
#   bash 03_run_experiments.sh table4     # svih pet varijanti, ViTDet boxes
#   bash 03_run_experiments.sh table5     # xl1, YOLOv8 + GroundingDINO boxes
#   bash 03_run_experiments.sh all        # sve, redosledno: validate, table3, table4, table5
# ---------------------------------------------------------------------------

set -uo pipefail   # namerno nije -e: jedan failed run ne treba da ubije ceo batch, samo da se zabeleži u logu i nastavi dalje.

WORKSPACE="/workspace"
REPO="${WORKSPACE}/efficientvit"
COCO="${WORKSPACE}/coco"
LOGS="${WORKSPACE}/logs"
EVAL="applications/efficientvit_sam/eval_efficientvit_sam_model.py"

mkdir -p "${LOGS}"

cd "${REPO}" || {
    echo "ERROR: repo not found at ${REPO}. Run 01_setup_pod.sh first."
    exit 1
}
[ -f "${EVAL}" ] || {
    echo "ERROR: ${EVAL} missing. Has the repo layout changed upstream?"
    exit 1
}

MODE="${1:-validate}"

# ---------------------------------------------------------------------------
run_eval () {
    local tag="$1"; shift
    local log="${LOGS}/${tag}.log"
    local tmp="${LOGS}/${tag}.running"

    if [ -f "${log}" ]; then
        echo ">>> ${tag}: already complete, skipping. (delete ${log} to re-run)"
        return 0
    fi

    echo ">>> ${tag}: starting at $(date '+%H:%M:%S')"
    local start=$SECONDS

    # Upisuje se u privremeno ime i jedino ide u ${tag}.log pri uspešnom završetku.
    # Inače bi crash-ovani run-ovi ostavljali logove za sobom i preskakali pri sledećem run-u.
    # Kada bi se neuspeli runovi ostavljali logove za sobom, preskakali bi se pri sledećem run-u i gledali bi se kao da su završeni — a zapravo bi se ti rezultati izgubili bez ikakve greške.

    torchrun --nproc_per_node=1 "${EVAL}" "$@" 2>&1 | tee "${tmp}"
    local status=${PIPESTATUS[0]}

    if [ "${status}" -ne 0 ]; then
        mv "${tmp}" "${LOGS}/${tag}.FAILED"
        echo ">>> ${tag}: FAILED (exit ${status})"
        echo ">>> log kept at ${LOGS}/${tag}.FAILED; the run will be retried next time"
        echo
        return 1
    fi

    echo ">>> ${tag}: finished in $(( SECONDS - start ))s" >> "${tmp}"
    mv "${tmp}" "${log}"
    echo ">>> ${tag}: done in $(( SECONDS - start ))s"
    echo
    return 0
}

gt_box () {   # $1 = model short name
    run_eval "table3_${1}_coco_gtbox" \
        --dataset coco \
        --image_root "${COCO}/val2017" \
        --annotation_json_file "${COCO}/annotations/instances_val2017.json" \
        --model "efficientvit-sam-${1}" \
        --prompt_type box
}

det_box () {  # $1 = model short name, $2 = detector name
    run_eval "detbox_${1}_coco_${2}" \
        --dataset coco \
        --image_root "${COCO}/val2017" \
        --annotation_json_file "${COCO}/annotations/instances_val2017.json" \
        --model "efficientvit-sam-${1}" \
        --prompt_type box_from_detector \
        --source_json_file "${COCO}/source_json_file/coco_${2}.json"
}

# ---------------------------------------------------------------------------
case "${MODE}" in

  validate)
    cat <<'EOF'
=============================================================
VALIDATION RUN — xl1, ground-truth boxes, COCO val2017

Expected:  all=79.927  large=83.748  medium=82.210  small=75.833

Ne nastaljati na sledeći run dok se ovi rezultati ne poklope. 
Ovaj jedan rezultat validira rukovanje rezolucijom, transformacije koordinata, 
dekodiranje maski i prosekovanje sve odjednom. 
Sve što ide dalje nasleđuje sve što je pogrešno ovde.
=============================================================
EOF
    gt_box xl1
    echo "=== result line from that run ==="
    grep -E "all=" "${LOGS}/table3_xl1_coco_gtbox.log" 2>/dev/null \
      || echo "(no result line found — check the log or the .FAILED file)"
    echo
    echo "Expected: all=79.927  large=83.748  medium=82.210  small=75.833"
    echo "Within ~0.1 counts as reproduced. Off by more than 1.0 means stop and debug."
    ;;

  table3)   # GT boxes. Samo xl1 imamo iz rada, ostalih 4 su novi brojevi.
    for m in xl1 l0 l1 l2 xl0; do gt_box "$m"; done
    ;;

  table4)   # ViTDet boxes. Svih 5 postoji u radu: 45.7 46.2 46.6 47.5 47.8
    for m in xl1 l0 l1 l2 xl0; do det_box "$m" vitdet; done
    ;;

  table5)   # Isti model, drugačiji tip detektora. Očekivani xl1: 44.7 / 48.2
    det_box xl1 yolov8
    det_box xl1 groundingdino
    ;;

  all)
    bash "$0" validate
    bash "$0" table3
    bash "$0" table4
    bash "$0" table5
    ;;

  *)
    echo "Unknown mode: ${MODE}"
    echo "Use one of: validate | table3 | table4 | table5 | all"
    exit 1
    ;;
esac

echo "Logs are in ${LOGS}/"
ls -la "${LOGS}/"
