#!/usr/bin/env python3
"""
marvel_dataset.py — loader for GRN_MARVEL, and a --stats mode.

The two things this exists to get right
---------------------------------------
1. CLASS IDS ARE PER-CLIP. The 19 clips were annotated as separate CVAT tasks
   and exported independently, so each has its own obj.names with the same
   seven classes in a DIFFERENT ORDER. There are 13 distinct orderings. One
   clip, "pedestrian_1 (7)", additionally has an eighth class ("Anomalous
   Object") at index 0, shifting every other id by one.

   Assuming a single global mapping silently mislabels tens of thousands of
   objects. Every clip's obj.names must be read individually.

2. FILENAMES DO NOT MATCH between video and annotation:
       video       pedestrian_1_(7).mp4
       annotation  pedestrian_1 (7)/
   and timestamps differ in whether "+" survived:
       video       ..._07-46-37+0000--...
       annotation  ..._07-46-370000--...
   Pairing needs a normalised key, not string equality.

Box format: YOLO — "class_id x_center y_center width height", all normalised
to [0,1] and centre-based. SAM wants absolute [x1,y1,x2,y2].

Usage
-----
    python 07_marvel_dataset.py --stats           # no video needed
    python 07_marvel_dataset.py --check           # verify video pairing
"""

import argparse
import collections
import os
import re
import sys

# Canonical label set. Everything is normalised into this so results are
# comparable across clips.
CANONICAL = [
    "car",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "heavy goods vehicle",
    "light goods vehicle",
    "bus",
    "anomalous object",   # only in pedestrian_1 (7); not a vehicle category
]

DEFAULT_ROOT = os.path.expanduser(
    "~/Documents/ml2/projekat/marvel/MULTIMODAL_DATASET"
)


def norm_label(s):
    """Lowercase and collapse whitespace: 'Pedestrian' -> 'pedestrian'."""
    return re.sub(r"\s+", " ", s.strip().lower())


def norm_key(s):
    """Key for pairing video files with annotation directories.

    Strips the extension and every character that differs between the two
    naming schemes: spaces, underscores, plus signs, parentheses, hyphens.
    """
    s = os.path.splitext(os.path.basename(s))[0]
    return re.sub(r"[^a-z0-9]", "", s.lower())


class Clip:
    def __init__(self, name, ann_dir, video_path=None):
        self.name = name
        self.ann_dir = ann_dir
        self.video_path = video_path

        names_file = os.path.join(ann_dir, "obj.names")
        if not os.path.isfile(names_file):
            raise FileNotFoundError(
                f"{name}: no obj.names in {ann_dir}. The clip archive was "
                "probably not fully extracted — see 02_download_data.sh."
            )
        with open(names_file) as f:
            raw = [ln for ln in (l.strip() for l in f) if ln]
        # index in this clip's file -> canonical label
        self.id_to_label = {i: norm_label(n) for i, n in enumerate(raw)}

        unknown = set(self.id_to_label.values()) - set(CANONICAL)
        if unknown:
            print(f"  WARNING: {self.name}: unrecognised labels {unknown}",
                  file=sys.stderr)

        data_dir = os.path.join(ann_dir, "obj_train_data")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"{name}: no obj_train_data/ in {ann_dir}. Incomplete extraction."
            )
        self.frame_files = sorted(
            f for f in os.listdir(data_dir)
            if f.startswith("frame_") and f.endswith(".txt")
        )

    def __len__(self):
        return len(self.frame_files)

    def read_frame(self, idx, width=None, height=None):
        """Return a list of (label, box) for one frame.

        box is [xc, yc, w, h] normalised if width/height are None,
        otherwise [x1, y1, x2, y2] in absolute pixels.
        """
        path = os.path.join(self.ann_dir, "obj_train_data", self.frame_files[idx])
        out = []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid = int(parts[0])
                xc, yc, w, h = (float(v) for v in parts[1:])
                label = self.id_to_label.get(cid, f"unknown_{cid}")

                if width is None or height is None:
                    out.append((label, [xc, yc, w, h]))
                else:
                    # centre+size normalised  ->  corner+corner absolute
                    out.append((label, [
                        (xc - w / 2) * width,
                        (yc - h / 2) * height,
                        (xc + w / 2) * width,
                        (yc + h / 2) * height,
                    ]))
        return out

    def frame_size(self):
        """Read dimensions from the video rather than assuming them.

        Tries OpenCV first (present on the pod), falls back to ffprobe
        (easier to have on a Mac), and explains itself if neither exists.
        """
        if self.video_path is None:
            raise RuntimeError(f"{self.name}: no video paired")

        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w and h:
                return w, h
        except ImportError:
            pass

        import json as _json
        import subprocess
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "json",
                 self.video_path],
                capture_output=True, text=True, timeout=30,
            )
            s = _json.loads(out.stdout)["streams"][0]
            return int(s["width"]), int(s["height"])
        except FileNotFoundError:
            raise RuntimeError(
                "Need either opencv-python or ffmpeg to read video dimensions.\n"
                "  pip3 install opencv-python     (also needed for frame decoding)\n"
                "  brew install ffmpeg            (lighter, if you only want sizes)"
            )


def discover(root):
    ann_root = os.path.join(root, "bbox", "bounding_box_annotation", "extracted")
    vid_root = os.path.join(root, "anonymised_videos")

    if not os.path.isdir(ann_root):
        sys.exit(f"No annotations at {ann_root}\n"
                 "Extract bounding_box_annotation.zip and its 19 inner zips first.")

    videos = {}
    if os.path.isdir(vid_root):
        for f in sorted(os.listdir(vid_root)):
            if not f.lower().endswith(".mp4"):
                continue
            k = norm_key(f)
            # Two videos normalising to the same key would make pairing
            # arbitrary — better to know than to silently use whichever
            # happened to be listed last.
            if k in videos:
                print(f"  WARNING: '{f}' and "
                      f"'{os.path.basename(videos[k])}' collide under "
                      f"norm_key('{k}')", file=sys.stderr)
            videos[k] = os.path.join(vid_root, f)

    clips = []
    for d in sorted(os.listdir(ann_root)):
        full = os.path.join(ann_root, d)
        if not os.path.isdir(full):
            continue
        clips.append(Clip(d, full, videos.get(norm_key(d))))
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    clips = discover(args.root)
    print(f"{len(clips)} clips, {sum(len(c) for c in clips):,} annotated frames\n")

    if args.check:
        print("=== video pairing ===")
        missing = [c for c in clips if c.video_path is None]
        for c in clips:
            mark = "ok " if c.video_path else "MISSING"
            print(f"  {mark}  {c.name}")
        if missing:
            print(f"\n{len(missing)} clips have no matching video. Extract "
                  "anonymised_videos.zip, or widen norm_key().")
        else:
            print("\nAll clips paired.")
            print("\n=== frame sizes ===")
            sizes = collections.Counter()
            for c in clips:
                try:
                    wh = c.frame_size()
                except RuntimeError as e:
                    print(e)
                    break
                sizes[wh] += 1
                print(f"  {wh[0]}x{wh[1]}  {c.name}")
            if len(sizes) == 1:
                print(f"\nAll clips share one resolution: "
                      f"{list(sizes)[0][0]}x{list(sizes)[0][1]}")
            elif sizes:
                print(f"\n{len(sizes)} different resolutions — the loader must "
                      "denormalise per clip, not with a global constant.")
        return

    if args.stats:
        print("=== class ordering per clip ===")
        orderings = collections.defaultdict(list)
        for c in clips:
            key = tuple(c.id_to_label[i] for i in sorted(c.id_to_label))
            orderings[key].append(c.name)
        print(f"{len(orderings)} distinct orderings across {len(clips)} clips\n")
        for key, names in sorted(orderings.items(), key=lambda kv: -len(kv[1])):
            print(f"  [{len(names)} clip(s)] " + " | ".join(key))

        print("\n=== TRUE class distribution (after per-clip remapping) ===")
        counts = collections.Counter()
        per_clip_frames = 0
        for c in clips:
            per_clip_frames += len(c)
            for i in range(len(c)):
                for label, _ in c.read_frame(i):
                    counts[label] += 1

        total = sum(counts.values())
        for label, n in counts.most_common():
            print(f"  {label:<24} {n:>8,}  ({100 * n / total:5.2f}%)")
        print(f"  {'TOTAL':<24} {total:>8,}")
        print(f"\n  {total / per_clip_frames:.2f} objects per frame")

        print("\nCompare against the naive count that ignores per-clip mappings —")
        print("the difference is how many objects would have been mislabelled.")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
