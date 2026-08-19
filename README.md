# EfficientViT-SAM on GRN MARVEL

Evaluation scripts reproducing the box-prompted zero-shot benchmark from
**EfficientViT-SAM: Accelerated Segment Anything Model Without Performance Loss**
(Zhang, Cai & Han, CVPRW 2024) on COCO, and applying the model to the
GRN_MARVEL traffic surveillance dataset.

- Paper: https://openaccess.thecvf.com/content/CVPR2024W/ELVM/html/Zhang_EfficientViT-SAM_Accelerated_Segment_Anything_Model_Without_Performance_Loss_CVPRW_2024_paper.html
- Model and official code: https://github.com/mit-han-lab/efficientvit
- Dataset: https://zenodo.org/records/10671777

The evaluation itself is the authors' `eval_efficientvit_sam_model.py`, run
unmodified. What lives here is the surrounding work: environment setup, data
acquisition, a reproducible run matrix, and three analyses their script does
not provide — predicted-versus-true IoU calibration, separated encoder and
decoder timing, and a loader plus evaluation for MARVEL.

Written for a single rented GPU (RunPod, RTX A5000, 24 GB) with a persistent
network volume mounted at `/workspace`.

---

## Running these

Run order on a fresh pod. Copy the contents of this folder to `/workspace/scripts/` once — it lives on the network volume from then on, so this is a one-time step.

```bash
bash /workspace/scripts/01_setup_pod.sh          # every session
bash /workspace/scripts/02_download_data.sh      # once, ever
bash /workspace/scripts/03_run_experiments.sh validate
```

## What each does

**`01_setup_pod.sh`** — verifies the volume is attached and a GPU is visible, clones the repo if it isn't already there, installs dependencies, prints an environment summary.

Re-run this on every new pod. Installed Python packages live on the pod's temporary disk and die with it; the repo lives on the volume and survives.

Copy the environment summary into the report — speed numbers are meaningless without the GPU name, CUDA version and torch version.

**`02_download_data.sh`** — COCO val2017, annotations, the three pre-generated detector box files, all five checkpoints, and GRN MARVEL. Roughly 4–5 GB, 10–20 minutes.

Runs once ever, since everything lands on the volume. Every step is skipped if the target already exists, so it's safe to re-run after an interrupted download.

**`03_run_experiments.sh`** — the item 3 matrix. Each run writes a timestamped log to `/workspace/logs/` and is skipped if that log already exists, so you can stop, terminate the pod, and resume another day.

| Mode | Runs | Expected |
|---|---|---|
| `validate` | xl1, GT boxes | `all=79.927 / L=83.748 / M=82.210 / S=75.833` |
| `table3` | all five, GT boxes | only xl1 published — the rest are new |
| `table4` | all five, ViTDet boxes | 45.7 / 46.2 / 46.6 / 47.5 / 47.8 |
| `table5` | xl1, YOLOv8 + GroundingDINO | 44.7 / 48.2 |
| `all` | everything in order | — |

## Before the full run

Time a subset first. Point the eval at ~100 images, measure, multiply by 50. Two minutes of work tells you your actual throughput and catches anything pathological before you commit to a full pass.

## Stop at validation

`validate` must match `79.927` before anything else runs. A pipeline that is subtly broken but produces plausible numbers is the worst outcome available — you'd only find out when the committee asks why L2 beats XL0.

If it doesn't match, likely causes in order:

1. **Checkpoint mismatch** — verify the file downloaded properly and isn't a truncated HTML error page (`ls -la` and check the size).
2. **Wrong image root** — the COCO dataset class lists files with `os.listdir(image_root)` and parses the image id from the filename, so a stray non-image file in `val2017/` breaks it.
3. **Wrong annotation file** — `instances_val2017.json`, not the train split.

Note two details of the official protocol, both of which differ from the obvious guess:

- It uses `multimask_output=True` and keeps whichever of the three candidates has the highest **predicted** IoU. The prediction head selects the mask that gets scored.
- It does **not** filter `iscrowd` annotations — it loads every annotation and skips only `area < 1`.

## The extra scripts

These do things the repo's eval script doesn't.

**`00_local_smoke_test.py`** — one image, one box, one mask. Detects CUDA/MPS/CPU automatically, so it works as the first check on a new pod as well as locally. Run it before anything heavier.

```bash
python 00_local_smoke_test.py some_photo.jpg
```

**`04_calibration_eval.py`** — logs the model's **predicted** IoU next to the **true** IoU for every instance, and reports bias, mean absolute error and correlation.

This is required groundwork, not an extra. MARVEL has no ground-truth masks, so predicted IoU is the only number available there — and you can only justify using it if you've shown on COCO that it tracks reality. COCO is the only place you have both.

```bash
python 04_calibration_eval.py --model efficientvit-sam-l0 --num-images 200   # quick
python 04_calibration_eval.py --model efficientvit-sam-xl1                   # full
```

Its mIoU may land slightly off 79.927 because it runs its own loop rather than the official one. That's fine — the headline number comes from `03`, and this script exists for the correlation.

**`05_benchmark_speed.py`** — encoder and decoder timed separately, with warm-up and median over 50 runs, plus `torch.cuda.synchronize()` so you measure the GPU rather than Python. Prints your numbers against the paper's and estimates how long a full COCO pass will take.

```bash
python 05_benchmark_speed.py --models l0 xl1
```

**`06_inspect_marvel.py`** — walks the extracted MARVEL directory and reports what's in it: file types, tree, the structure of any JSON/XML/CSV, and video resolution and frame counts via `ffprobe`. Makes no assumptions about the schema.

No GPU needed — run this on your laptop as soon as the zip is extracted. The annotation format is the biggest remaining unknown in item 4.

```bash
python 06_inspect_marvel.py /path/to/extracted/marvel
```

## Still not written, deliberately

Log parsing, the results table, and the MARVEL dataloader all depend on formats nobody has seen yet. Writing them against guesses produces code you throw away. They become quick once you have one real log and the output of `06`.

## Housekeeping

- Work only inside `/workspace`. Anything written to `/root`, `/tmp` or the home directory dies with the pod.
- Terminate the pod when done for the day — not just stop it. The volume holds everything.
- Record `git rev-parse --short HEAD` from the repo; the task requires stating which code version was used.
