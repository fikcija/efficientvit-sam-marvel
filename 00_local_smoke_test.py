"""
00_local_smoke_test.py — pokretanje EfficientViT-SAM lokalno, na jednoj slici.

Šta radi: koristimo za potvrdu razumevanja API-ja i da model radi pre nego što se iznajmi GPU.
Ša ne radi: Ovo ne reprodukuje originalni rad i vremena zabeležena u radu. 

Setup (macOS, Apple Silicon):

    python3 -m venv .venv && source .venv/bin/activate
    pip install torch torchvision numpy pillow matplotlib timm
    git clone https://github.com/mit-han-lab/efficientvit.git

Pokrenuti unutar kloniranog repozitorijuma da se resolv-uju importi:

    cd efficientvit
    PYTORCH_ENABLE_MPS_FALLBACK=1 python /path/to/00_local_smoke_test.py IMAGE.jpg

"""

import os
import sys
import time

import numpy as np
import torch
from PIL import Image

# --- device ----------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"torch  : {torch.__version__}")
print(f"device : {DEVICE}")
if DEVICE == "cpu":
    print("  (no GPU acceleration — expect this to be slow but it should still work)")
print()

# --- repo na import putanji ------------------------------------------------
# Kada se skripta pokrene kao "python /putanja/do/skripte.py", Python na
# sys.path stavlja direktorijum SKRIPTE, a ne trenutni radni direktorijum.
# Pošto ove skripte žive van efficientvit repoa, cd u repo nije dovoljan —
# putanja se mora dodati eksplicitno.
_REPO = os.environ.get("EFFICIENTVIT_ROOT", "/workspace/efficientvit")
if os.path.isdir(_REPO) and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# --- model -----------------------------------------------------------------
try:
    from efficientvit.sam_model_zoo import create_efficientvit_sam_model
    from efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
except ImportError:
    sys.exit(
        f"Ne mogu da importujem efficientvit (tražen u {_REPO}).\n"
        "Ako je repo negde drugde:  export EFFICIENTVIT_ROOT=/putanja/do/efficientvit"
    )

# l0: 34.8M parametara, 35 GMACs, input dimeznacoja 512x512.
# pretrained=True za. download checkpoint-a.
print("Loading efficientvit-sam-l0 ...")
model = create_efficientvit_sam_model(name="efficientvit-sam-l0", pretrained=True)
model = model.to(DEVICE).eval()
predictor = EfficientViTSamPredictor(model)
print("Loaded.\n")

# --- image -----------------------------------------------------------------
image_path = sys.argv[1] if len(sys.argv) > 1 else None
if image_path is None:
    sys.exit("Usage: python 00_local_smoke_test.py IMAGE.jpg")

image = np.array(Image.open(image_path).convert("RGB"))
h, w = image.shape[:2]
print(f"Image: {image_path}  ({w}x{h})")

# --- encode ----------------------------------------------------------------
# set_image() pokreće image encoder. Ovo je najskuplji deo, i poenta je da se pokreće samo jednom po slici, a ne po promptu.
t0 = time.time()
predictor.set_image(image)
t_encode = time.time() - t0
print(f"Encoder: {t_encode * 1000:.0f} ms")

# --- prompt ----------------------------------------------------------------
# Random deo slike koji sadrži srednji deo slike. Ovo korisitmo samo za proveru pipline-a
# tako da konkretna maska nije bitna. 

box = np.array([w * 0.25, h * 0.25, w * 0.75, h * 0.75])
print(f"Box prompt: {box.round().astype(int).tolist()}")

# multimask_output=False jer su boxov prompti dovoljno precizni da ne treba da se generiše više maski.
t0 = time.time()
masks, iou_predictions, _ = predictor.predict(box=box, multimask_output=False)
t_decode = time.time() - t0
print(f"Decoder: {t_decode * 1000:.0f} ms")
print()

mask = masks[0]
print(f"Mask shape      : {mask.shape}  (should match the image: {h}x{w})")
print(f"Mask pixels     : {int(mask.sum()):,} of {h * w:,} "
      f"({100 * mask.sum() / (h * w):.1f}% of the image)")
print(f"Predicted IoU   : {float(iou_predictions[0]):.3f}")
print()
print("Predicted IoU - procena kvaliteta maske.")

# --- save a picture --------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    overlay = image.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([80, 70, 220])).astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].imshow(image); ax[0].set_title("input")
    ax[1].imshow(overlay); ax[1].set_title(f"mask (predicted IoU {float(iou_predictions[0]):.2f})")
    for a in ax:
        a.axis("off")
    x1, y1, x2, y2 = box
    ax[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  fill=False, edgecolor="yellow", lw=1.5))
    plt.tight_layout()
    plt.savefig("smoke_test_result.png", dpi=110)
    print("Wrote smoke_test_result.png")
except ImportError:
    print("(matplotlib not installed — bez vizualizacije)")

# ---------------------------------------------------------------------------
# Vremena se ne pokplapaju sa originalnim radom jer se u radu koristi
# fp16 na A100 sa 16 slika po batch-u.
# ---------------------------------------------------------------------------
