#!/usr/bin/env python3
"""
08_marvel_eval.py — pokrenuti EfficientViT-SAM nad GRN_MARVEL. Ovo je zadatak 4 u projektu.

Ključni problemi:
-------------------
MARVEL ima bounding boxes ali NE ground-truth maske, pa se mIoU ne može
izračunati. To je osobina skupa podataka, a ne greška. Šta ova
skripta beleži umesto toga:

  predicted IoU   procena kvaliteta modela. Samo se može verovati do
                  stepena koji je 04_calibration_eval.py pokazao da prati realnost na
                  COCO datasetu — pokrenuti prvo i citirati korelaciju.

  fill ratio      mask area / prompt box area. 
                  Dobra maska popunjva dobar deo svog bounding box-a. 
                  Blizu 0 znači da je model našao skoro ništa; blizu 1.0 znači da je vratio ceo bounding boc, što
                  obično znači da nije uspeo da pronađe granicu objekta.

  containment     udeo piksela maski koji leže unutar prompt bounding box-a. 
                  Trebalo bi da bude blizu 1. Niske vrednosti znače da maska izlazi van
                  prompta, što je jasan znak neuspeha.

Ni jedna od ovih nije mIoU. Zajedno nam one omogućavaju da kažemo nešto kvantitativno o
kvalitetu maske bez ground truth-a, što je realna pozicija iz koje se može raspravljati.

Uticaj parametara (konkretan zadatak 4 u projektu)
-------------------------------------------------
  --model      l0 / l1 / l2 / xl0 / xl1. L verzija modela koristi 512px input, a XL 1024px.
               MARVEL frejmovi su 1920x1080, tako da se objekat smanjuje na 0.27x svoje originalne veličine na
               512 za razliku od 0.53x at 1024 — ovde se očekuje da rezolucija ima veći uticaj nego
               što se može zaključiti iz Tabele 1 u radu.
  --jitter     skilaranje svakog boxa oko njegovog centra, npr. 0.1 = 10% veći box,
               -0.1 = 10% manji box. Meri osetljivost na kvalitet prompta,
               sintetski ekvivalent upotrebe sa stvarnim detektorima.
  --per-clip   izaberi N frejmova po klipu umesto korišćenja svih iz klipa. uzorkuje N frejmova po snimku 
               umesto svih. Koristiti za ispitivanje uticaja parametara: tri snimka sadrže 62% svih
               frejmova, pa bi uniformno uzorkovanje po celom skupu bilo pristrasno ka njima.

Korišćenje:
-----
    python 08_marvel_eval.py --model efficientvit-sam-xl1 --per-clip 100
    python 08_marvel_eval.py --model efficientvit-sam-l0  --jitter 0.1
    python 08_marvel_eval.py --model efficientvit-sam-xl1          # svaki frame

Pokrenuti unutar kloniranog repozitorijuma kako bi se importi rezolovali.
"""

import argparse
import collections
import csv
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marvel_dataset import discover  # noqa: E402

# Python na sys.path stavlja direktorijum SKRIPTE, a ne radni direktorijum,
# pa se putanja do efficientvit repoa mora dodati eksplicitno.
_REPO = os.environ.get("EFFICIENTVIT_ROOT", "/workspace/efficientvit")
if os.path.isdir(_REPO) and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from efficientvit.sam_model_zoo import create_efficientvit_sam_model  # noqa: E402
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor  # noqa: E402


def clip_box(box, W, H):
    """Ograničava okvir na dimenzije frejma.

    Primenjuje se na svaki okvir, ne samo na perturbovane. CVAT export ponekad
    sadrži normalizovane koordinate malo izvan [0,1] za objekte koji dodiruju
    ivicu frejma, što nakon denormalizacije daje negativne piksele. Negativan
    indeks u Pythonu znači brojanje od kraja niza, pa bi isečak za računanje
    obuhvaćenosti zahvatio pogrešan deo maske i dao tiho pogrešan broj umesto
    greške.
    """
    x1, y1, x2, y2 = box
    return [
        min(max(0.0, x1), float(W)), min(max(0.0, y1), float(H)),
        min(max(0.0, x2), float(W)), min(max(0.0, y2), float(H)),
    ]


def jitter_box(box, frac, W, H):
    """Skaliraj box oko njegovog centra po (1 + frac)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * (1 + frac), (y2 - y1) * (1 + frac)
    return clip_box([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], W, H)


def pick_indices(n_ann, n_img, per_clip):
    """Indeksi frejmova koji postoje i u anotacijama i među dekodovanim frejmovima."""
    usable = min(n_ann, n_img)
    if per_clip and per_clip < usable:
        # ravnomerno uzorkuj po klipu, a ne samo prvih N frejmova
        return np.linspace(0, usable - 1, per_clip).astype(int).tolist()
    return list(range(usable))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientvit-sam-xl1")
    ap.add_argument("--root", default="/workspace/marvel/MULTIMODAL_DATASET")
    ap.add_argument("--per-clip", type=int, default=0,
                    help="frames per clip; 0 = all")
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="box scale perturbation, e.g. 0.1 or -0.1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device; this will be extremely slow.\n")

    tag = args.model.replace("efficientvit-sam-", "")
    if args.jitter:
        tag += f"_jitter{args.jitter:+.2f}"
    out_path = args.out or f"/workspace/logs/marvel_{tag}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"model  : {args.model}")
    print(f"device : {device}")
    print(f"jitter : {args.jitter:+.2f}" if args.jitter else "jitter : none")

    model = create_efficientvit_sam_model(name=args.model, pretrained=True)
    predictor = EfficientViTSamPredictor(model.to(device).eval())

    clips = discover(args.root)
    frames_root = os.path.join(args.root, "frames")
    if not os.path.isdir(frames_root):
        sys.exit(f"No extracted frames at {frames_root}\n"
                 "Run 07_extract_marvel_frames.sh first.")

    rows = []
    t_start = time.time()
    n_frames = n_objects = 0

    for clip in clips:
        if clip.video_path is None:
            print(f"  SKIP {clip.name}: no video paired — run "
                  "marvel_dataset.py --check to diagnose")
            continue

        base = os.path.splitext(os.path.basename(clip.video_path))[0]
        frame_dir = os.path.join(frames_root, base)
        if not os.path.isdir(frame_dir):
            print(f"  SKIP {clip.name}: no frames at {frame_dir}")
            continue

        n_img = len([f for f in os.listdir(frame_dir) if f.endswith(".jpg")])
        indices = pick_indices(len(clip), n_img, args.per_clip)
        print(f"  {clip.name}: {len(indices)} frames")

        for idx in indices:
            jpg = os.path.join(frame_dir, f"frame_{idx:06d}.jpg")
            if not os.path.exists(jpg):
                continue

            image = np.array(Image.open(jpg).convert("RGB"))
            H, W = image.shape[:2]

            objs = clip.read_frame(idx, width=W, height=H)
            if not objs:
                continue

            t0 = time.time()
            predictor.set_image(image)          # encoder, jednom po frejmu, ne po promptu
            t_enc = time.time() - t0
            n_frames += 1

            for label, box in objs:
                box = (jitter_box(box, args.jitter, W, H) if args.jitter
                       else clip_box(box, W, H))
                x1, y1, x2, y2 = box
                box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if box_area < 1:
                    continue

                t0 = time.time()
                masks, iou_pred, _ = predictor.predict(
                    box=np.array(box), multimask_output=True)
                t_dec = time.time() - t0

                # Same selection rule as the official COCO protocol.
                best = int(iou_pred.argmax())
                mask = masks[best].astype(bool)

                mask_area = int(mask.sum())
                inside = int(mask[int(y1):int(y2), int(x1):int(x2)].sum())

                rows.append({
                    "clip": clip.name,
                    "frame": idx,
                    "label": label,
                    "box_area_px": round(box_area, 1),
                    # šta enkoder vidi na 512 (L) i 1024 (XL), tj. koliki je box kada se frejm smanji na 512 ili 1024
                    "box_area_at_512": round(box_area * (512 / max(W, H)) ** 2, 1),
                    "box_area_at_1024": round(box_area * (1024 / max(W, H)) ** 2, 1),
                    "mask_area_px": mask_area,
                    "pred_iou": round(float(iou_pred[best]), 5),
                    "fill_ratio": round(mask_area / box_area, 4),
                    "containment": round(inside / mask_area, 4) if mask_area else 0.0,
                    "chosen_mask": best,
                    "encoder_ms": round(t_enc * 1000, 2),
                    "decoder_ms": round(t_dec * 1000, 2),
                })
                n_objects += 1

        elapsed = time.time() - t_start
        print(f"    running total: {n_frames} frames, {n_objects} objects, "
              f"{elapsed / 60:.1f} min")

    if not rows:
        sys.exit("No results produced — check the frame directory names.")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows):,} rows to {out_path}")

    # --- summary -----------------------------------------------------------
    by_label = collections.defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    print(f"\n=== {args.model}"
          + (f", jitter {args.jitter:+.2f}" if args.jitter else "") + " ===")
    print(f"{'class':<24}{'n':>8}{'pred IoU':>10}{'fill':>8}"
          f"{'contain':>9}{'box px':>10}")
    for label, rs in sorted(by_label.items(), key=lambda kv: -len(kv[1])):
        print(f"{label:<24}{len(rs):>8}"
              f"{np.mean([r['pred_iou'] for r in rs]):>10.4f}"
              f"{np.mean([r['fill_ratio'] for r in rs]):>8.3f}"
              f"{np.mean([r['containment'] for r in rs]):>9.3f}"
              f"{np.mean([r['box_area_px'] for r in rs]):>10.0f}")

    print(f"\n{'ALL':<24}{len(rows):>8}"
          f"{np.mean([r['pred_iou'] for r in rows]):>10.4f}"
          f"{np.mean([r['fill_ratio'] for r in rows]):>8.3f}"
          f"{np.mean([r['containment'] for r in rows]):>9.3f}")

    print(f"\nencoder median {np.median([r['encoder_ms'] for r in rows]):.1f} ms/frame")
    print(f"decoder median {np.median([r['decoder_ms'] for r in rows]):.1f} ms/object")
    print(f"total {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
