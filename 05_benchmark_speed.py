#!/usr/bin/env python3
"""
05_benchmark_speed.py — merenje brzine enkodera i dekodera.

Zašto odvojedno merenje:
--------------
U radu je zabeležena proposnost (762 img/s for L0, 182 for XL1) koja predstavlja propusnost
ENCODER-a. Reportovanje jednog kombinovanog broja ne bi bilo uporedivo. Ova
skripta vremenski meri dva koraka nezavisno, kako bi tabele bile u skladu sa Tabelom 1.

Dve stvari koje su lako pogrešno izmeriti i koje ova skripta rešava:

  warm-up   — prvi forward pasovi uključuju CUDA context setup, kernel
              autotuning i memory allocation, i GPU nije dostigao
              svoj steady clock. Ove iteracije mogu biti nekoliko puta sporije u realnosti, pa se discarduju.

  sync      — CUDA je asinhrona. Bez torch.cuda.synchronize() pre čitanja sata, 
              merenje vremena u Pythonu meri koliko brzo Python šalje komande na GPU, 
              a ne koliko brzo GPU izvršava.

Korišćenje:
-----
    python 05_benchmark_speed.py                       # svih 5 varijanti, 50 iteracija, 15 warmup
    python 05_benchmark_speed.py --models l0 xl1
    python 05_benchmark_speed.py --iters 100

Pokrenuti unutar kloniranog repoa. 
"""

import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np
import torch

# Python na sys.path stavlja direktorijum SKRIPTE, a ne radni direktorijum,
# pa se putanja do efficientvit repoa mora dodati eksplicitno.
_REPO = os.environ.get("EFFICIENTVIT_ROOT", "/workspace/efficientvit")
if os.path.isdir(_REPO) and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from efficientvit.sam_model_zoo import create_efficientvit_sam_model  # noqa: E402
from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor  # noqa: E402

# U Tabeli 1 — A100, TensorRT, fp16, batch 16. Naši rezultati će biti sporiji jer ne koristimo TensorRT niti fp16, i batch size je 1.;
# Stoje tu da bi ta razlika bila jasna, a ne da bi se reprodukovali. 
PAPER = {"l0": 762, "l1": 638, "l2": 538, "xl0": 278, "xl1": 182}
RESOLUTION = {"l0": 512, "l1": 512, "l2": 512, "xl0": 1024, "xl1": 1024}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_median(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    sync()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["l0", "l1", "l2", "xl0", "xl1"])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--out", default="/workspace/logs/speed.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device. Timings will not be meaningful.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    print(f"device : {gpu}")
    print(f"torch  : {torch.__version__}   cuda: {torch.version.cuda}")
    print(f"iters  : {args.iters} (median), warmup {args.warmup}\n")

    # Uvek ista sintetička slika. Content ne utiče na vreme merenja, a korišćenje
    # istog ulaza za svaki model čini poređenje čistijim.
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8)
    box = np.array([256, 256, 768, 768])

    results = []
    for short in args.models:
        name = f"efficientvit-sam-{short}"
        print(f"--- {name} ({RESOLUTION[short]}x{RESOLUTION[short]}) ---")

        model = create_efficientvit_sam_model(name=name, pretrained=True)
        predictor = EfficientViTSamPredictor(model.to(device).eval())

        with torch.no_grad():
            t_enc = time_median(lambda: predictor.set_image(image),
                                args.iters, args.warmup)

            predictor.set_image(image)   # embedding mora da postoji pre predict-a
            t_dec = time_median(
                lambda: predictor.predict(box=box, multimask_output=False),
                args.iters, args.warmup,
            )

        row = {
            "model": name,
            "resolution": RESOLUTION[short],
            "encoder_ms": round(t_enc * 1000, 2),
            "decoder_ms": round(t_dec * 1000, 2),
            "encoder_img_per_s": round(1 / t_enc, 1),
            "paper_a100_trt_img_per_s": PAPER[short],
            "gpu": gpu,
        }
        results.append(row)

        print(f"  encoder : {row['encoder_ms']:>8.2f} ms   "
              f"({row['encoder_img_per_s']:.1f} img/s)")
        print(f"  decoder : {row['decoder_ms']:>8.2f} ms   per prompt")
        print(f"  paper   : {PAPER[short]} img/s (A100 + TensorRT fp16, bs16)")
        print(f"  ratio   : {row['encoder_img_per_s'] / PAPER[short]:.2f}x of paper\n")

        del model, predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {args.out}")

    # Estimacija koliko dugo će trajati pun COCO pass, tako da znamo pre nego što krenemo sa obradom.
    print("\nEstimated full COCO val2017 pass (5000 images, ~36k instances):")
    for r in results:
        est = (5000 * r["encoder_ms"] + 36000 * r["decoder_ms"]) / 1000 / 60
        print(f"  {r['model']:<26} ~{est:.1f} min  (excluding image loading)")


if __name__ == "__main__":
    main()
