#!/usr/bin/env python3
"""
04_calibration_eval.py — loguje prediktovane IoU pored pravih IoU na COCO datasetu.

Zašto je ovo važno:
---------------
Eval skripta iz repozitorijuma izveštava mIoU ali odbacuje modelove sopstvene
prediktovane-IoU skorove. Trebaju nam i oni, jer MARVEL nema ground-truth maske: 
tamo je prediktovani IoU jedini numerički signal koji je dostupan.

Pre nego što se možemo osloniti na njega, moramo da pokažemo da je pouzdan. 
COCO je jedino mesto gde imamo oba broja, pa ih ova skripta sakuplja u parovima 
i meri koliko se slažu.

Šta ovo nije:
--------------
Ovo nije reprodukcija Tabele 3. 
Pokreće sopstvenu petlju kroz predictor API, tako danjegova predprocesiranja 
može da se razlikuje od zvanične skripte i njegov mIoU može da se malo razlikuje od 79.927. 
To je i očekivano. Glavni broj dolazi iz 03_run_experiments.sh. 
Ova skripta postoji zbog KORELACIJE između prediktovanog i pravog IoU.

Korišćenje:
-----
    python 04_calibration_eval.py --model efficientvit-sam-l0 --num-images 200
    python 04_calibration_eval.py --model efficientvit-sam-xl1   # all 5000

Pokrenuti unutar kloniranog repozitorijuma kako bi se importi rezolovali.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO

# Python na sys.path stavlja direktorijum SKRIPTE, a ne radni direktorijum,
# pa se putanja do efficientvit repoa mora dodati eksplicitno.
_REPO = os.environ.get("EFFICIENTVIT_ROOT", "/workspace/efficientvit")
if os.path.isdir(_REPO):
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    # Putanje do checkpoint-a su relativne u odnosu na radni direktorijum,
    # pa repo mora biti cwd. Sve putanje do podataka ovde su apsolutne.
    os.chdir(_REPO)

from efficientvit.sam_model_zoo import create_efficientvit_sam_model  # noqa: E402
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor  # noqa: E402

# COCO area thresholds: small < 32^2, large > 96^2
SMALL, LARGE = 32 ** 2, 96 ** 2


def bucket(area):
    if area < SMALL:
        return "small"
    if area < LARGE:
        return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientvit-sam-xl1")
    ap.add_argument("--image-root", default="/workspace/coco/val2017")
    ap.add_argument("--annotations",
                    default="/workspace/coco/annotations/instances_val2017.json")
    ap.add_argument("--num-images", type=int, default=0,
                    help="0 = all 5000. Use a few hundred for a quick check.")
    ap.add_argument("--out", default="/workspace/logs/calibration.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}   model: {args.model}")

    model = create_efficientvit_sam_model(name=args.model, pretrained=True)
    predictor = EfficientViTSamPredictor(model.to(device).eval())

    coco = COCO(args.annotations)
    img_ids = sorted(coco.getImgIds())
    if args.num_images:
        img_ids = img_ids[:args.num_images]
    print(f"images: {len(img_ids)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    t_start = time.time()

    for n, img_id in enumerate(img_ids, 1):
        info = coco.loadImgs(img_id)[0]
        image = np.array(
            Image.open(os.path.join(args.image_root, info["file_name"])).convert("RGB")
        )

        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=img_id))
                if a["area"] >= 1]
        if not anns:
            continue

        # Enkoder se pokreće jednom po slici, izvan annotation loop-a. 
        # Ovo je cela poenta SAM arhitekture — skupi deo se raspodeljuje na svaki prompt za tu sliku.
        predictor.set_image(image)

        for ann in anns:
            # COCO čuva [x, y, width, height]; model očekuje [x1,y1,x2,y2].
            x, y, w, h = ann["bbox"]
            box = np.array([x, y, x + w, y + h])

            # multimask_output=True kako vi se poklopilo sa oficijalnim protokolom: 
            # generišu se sva tri kandidata, a onda se zadržava onaj koji model ocenjuje najviše.
            # IoU head nije tu bez razloga, on bira masku koja se ocenjuje. 
            masks, iou_pred, _ = predictor.predict(box=box, multimask_output=True)
            best = int(iou_pred.argmax())

            pred = masks[best].astype(bool)
            gt = coco.annToMask(ann).astype(bool)

            union = np.logical_or(pred, gt).sum()
            if union == 0:
                continue
            true_iou = np.logical_and(pred, gt).sum() / union

            rows.append({
                "image_id": img_id,
                "ann_id": ann["id"],
                "category_id": ann["category_id"],
                "area": ann["area"],
                "size": bucket(ann["area"]),
                "true_iou": round(float(true_iou), 5),
                "pred_iou": round(float(iou_pred[best]), 5),
                "chosen_mask": best,
            })

        if n % 100 == 0:
            rate = n / (time.time() - t_start)
            print(f"  {n}/{len(img_ids)} images  "
                  f"{len(rows)} instances  {rate:.1f} img/s")

    # --- write -------------------------------------------------------------
    if not rows:
        sys.exit("No instances processed. Check --image-root and --annotations.")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    # --- summary -----------------------------------------------------------
    true = np.array([r["true_iou"] for r in rows])
    pred = np.array([r["pred_iou"] for r in rows])
    sizes = np.array([r["size"] for r in rows])

    print("\n--- mIoU (mean over INSTANCES, not classes, not images) ---")
    print(f"  all    : {100 * true.mean():.3f}   (n={len(true)})")
    for b in ("small", "medium", "large"):
        m = sizes == b
        if m.any():
            print(f"  {b:<7}: {100 * true[m].mean():.3f}   (n={m.sum()})")

    print("\n--- calibration of the predicted-IoU head ---")
    print(f"  mean predicted : {pred.mean():.4f}")
    print(f"  mean true      : {true.mean():.4f}")
    print(f"  bias           : {pred.mean() - true.mean():+.4f}"
          "   (positive = model is over-confident)")
    print(f"  mean abs error : {np.abs(pred - true).mean():.4f}")
    print(f"  Pearson r      : {np.corrcoef(pred, true)[0, 1]:.4f}")
    print()
    print("A high correlation means predicted IoU tracks real quality, which")
    print("justifies using it as a proxy metric on MARVEL. A low one is still")
    print("a finding — it just means it as a limitation instead.")


if __name__ == "__main__":
    main()
