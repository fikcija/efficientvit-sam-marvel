#!/usr/bin/env python3
"""
09_visualize_marvel.py — nacrtaj maske preko frejmova, po zadatom kriterijumu.

Čemu služi
----------
Na MARVEL-u nema tačnih maski, pa se kvalitet ne može izmeriti direktno. Ali
može se POGLEDATI. Metrike biraju ŠTA gledati, oči objašnjavaju ZAŠTO je nešto
loše — okluzija, zamućenje od kretanja, dva objekta u jednom okviru. Nijedan
broj to ne kaže.

Ovo je i izvor slika za izveštaj: zahtev je bar dva neuspešna slučaja.

Ne pokušavati da se odokativno rangira pet varijanti modela — razlika je par
poena mIoU i nije vidljiva golim okom. Za rangiranje služe metrike.

Kako radi
---------
Čita CSV koji je napravio 08_marvel_eval.py, izabere redove po kriterijumu,
ponovo pusti model SAMO na te frejmove i nacrta rezultat. Okvir se rekonstruiše
iz kolona x1,y1,x2,y2 — dakle tačno isti prompt kao u evaluaciji.

Primeri
-------
    # najgori slučajevi po popunjenosti — kandidati za "failure cases"
    python 09_visualize_marvel.py --csv /workspace/logs/marvel_xl1.csv \
        --mode worst --metric fill_ratio --n 8

    # najgori pešaci (mali objekti, gde je model najslabiji)
    python 09_visualize_marvel.py --csv /workspace/logs/marvel_xl1.csv \
        --mode worst --metric containment --label pedestrian --n 6

    # nasumičan presek, da se vidi tipično ponašanje
    python 09_visualize_marvel.py --csv /workspace/logs/marvel_xl1.csv \
        --mode random --n 8
"""

import argparse
import csv as csvmod
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_REPO = os.environ.get("EFFICIENTVIT_ROOT", "/workspace/efficientvit")
if os.path.isdir(_REPO):
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    os.chdir(_REPO)

from efficientvit.sam_model_zoo import create_efficientvit_sam_model  # noqa: E402
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default=None,
                    help="podrazumevano se izvodi iz imena CSV fajla")
    ap.add_argument("--root", default="/workspace/marvel/MULTIMODAL_DATASET")
    ap.add_argument("--mode", choices=["worst", "best", "random"], default="worst")
    ap.add_argument("--metric", default="fill_ratio",
                    choices=["fill_ratio", "containment", "pred_iou"])
    ap.add_argument("--label", default=None, help="filtriraj po klasi")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="/workspace/logs/vis")
    args = ap.parse_args()

    rows = list(csvmod.DictReader(open(args.csv)))
    if "x1" not in rows[0]:
        sys.exit("CSV nema kolone x1..y2 — napravljen je starijom verzijom "
                 "08_marvel_eval.py. Ponoviti evaluaciju.")

    if args.label:
        rows = [r for r in rows if r["label"] == args.label]
        if not rows:
            sys.exit(f"Nema redova sa klasom '{args.label}'")

    if args.mode == "random":
        random.seed(0)
        sel = random.sample(rows, min(args.n, len(rows)))
    else:
        rows.sort(key=lambda r: float(r[args.metric]),
                  reverse=(args.mode == "best"))
        sel = rows[:args.n]

    model_name = args.model or (
        "efficientvit-sam-"
        + os.path.basename(args.csv).replace("marvel_", "").split("_")[0].replace(".csv", "")
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model: {model_name}   device: {device}   primeraka: {len(sel)}")

    model = create_efficientvit_sam_model(name=model_name, pretrained=True)
    predictor = EfficientViTSamPredictor(model.to(device).eval())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    frames_root = os.path.join(args.root, "frames")

    ncol = 2
    nrow = (len(sel) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, sel):
        # Ime foldera sa frejmovima izvedeno je iz imena videa, koje se ne
        # poklapa sa imenom klipa iz anotacija — traži se normalizovano.
        cand = [d for d in os.listdir(frames_root)
                if "".join(ch for ch in d.lower() if ch.isalnum())
                == "".join(ch for ch in r["clip"].lower() if ch.isalnum())]
        if not cand:
            ax.set_title(f"nema frejmova: {r['clip'][:30]}")
            ax.axis("off")
            continue

        jpg = os.path.join(frames_root, cand[0], f"frame_{int(r['frame']):06d}.jpg")
        image = np.array(Image.open(jpg).convert("RGB"))
        box = np.array([float(r["x1"]), float(r["y1"]),
                        float(r["x2"]), float(r["y2"])])

        predictor.set_image(image)
        masks, iou_pred, _ = predictor.predict(box=box, multimask_output=True)
        mask = masks[int(iou_pred.argmax())].astype(bool)

        overlay = image.copy()
        overlay[mask] = (0.45 * overlay[mask]
                         + 0.55 * np.array([80, 70, 220])).astype(np.uint8)
        ax.imshow(overlay)
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                   fill=False, edgecolor="yellow", lw=1.8))
        ax.set_title(
            f"{r['label']}  |  fill {float(r['fill_ratio']):.2f}  "
            f"obuhv {float(r['containment']):.2f}  pred IoU {float(r['pred_iou']):.2f}\n"
            f"{r['clip'][:42]}  frejm {r['frame']}",
            fontsize=8)
        ax.axis("off")

    for ax in axes[len(sel):]:
        ax.axis("off")

    tag = f"{args.mode}_{args.metric}" + (f"_{args.label.replace(' ', '_')}" if args.label else "")
    out = os.path.join(args.out, f"{os.path.basename(args.csv)[:-4]}_{tag}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    print(f"Sačuvano: {out}")

    print("\nŽuti pravougaonik je prompt, ljubičasto je maska koju je model vratio.")
    print("Za izveštaj: uporediti šta je traženo i šta je dobijeno, i opisati")
    print("zašto je slučaj težak — to je ono što broj ne kaže.")


if __name__ == "__main__":
    main()
