#!/usr/bin/env python3
"""
06_inspect_marvel.py — nadji šta je zapravo unutar GRN_MARVEL.

Korišćeno da bi se orderila struktura dataset-a pre nego što se napiše dataloader
— kako su arhivirani podaci nestovani, CVAT YOLO format anotacije podataka, i redosled klasa po kliovima. 
Ovi pronalazi su iskorišćeni za pisanje marvel_dataset.py.

Zadržano jer ne pravi pretpostavke o shemi dataset-a, i može se koristiti da se potvrdi da 
je ekstrakcija kompletna, ili da se ispita druga verzija dataset-a.

Ne zahteva GPU. Pokrenuti na laptopu čim se zip ekstrahuje.

Korišćenje:
-----
    python 06_inspect_marvel.py /path/to/extracted/marvel
"""

import collections
import json
import os
import subprocess
import sys
import zipfile

TEXTLIKE = {".json", ".xml", ".csv", ".txt", ".yaml", ".yml", ".srt"}
VIDEO = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".mpg", ".mpeg"}
IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
AUDIO = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
ARCHIVE = {".zip"}


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def describe_json(path, indent="    "):
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:
        # Loše formatiran JSON je čest u istraživačkim skupovima podataka (trailing commas,
        # concatenated objects, JSONL). Pokaži sirovi tekst kako bi se mogao ručno pročitati
        # umesto da se odustane.
        print(f"{indent}(not valid JSON: {e})")
        print(f"{indent}--- raw contents ---")
        describe_text(path, indent, lines=60)
        return

    def walk(obj, prefix="", depth=0):
        pad = indent + "  " * depth
        if depth > 3:
            print(f"{pad}...")
            return
        if isinstance(obj, dict):
            print(f"{pad}{prefix}dict with {len(obj)} keys: {list(obj)[:12]}")
            for k, v in list(obj.items())[:5]:
                walk(v, f"{k}: ", depth + 1)
        elif isinstance(obj, list):
            print(f"{pad}{prefix}list of {len(obj)}")
            if obj:
                walk(obj[0], "[0]: ", depth + 1)
        else:
            val = repr(obj)
            print(f"{pad}{prefix}{type(obj).__name__} = "
                  f"{val[:70]}{'...' if len(val) > 70 else ''}")

    walk(data)


def describe_text(path, indent="    ", lines=8):
    try:
        with open(path, errors="replace") as f:
            for i, line in enumerate(f):
                if i >= lines:
                    print(f"{indent}...")
                    break
                print(f"{indent}{line.rstrip()[:110]}")
    except Exception as e:
        print(f"{indent}(could not read: {e})")


def describe_zip(path, indent="    ", limit=25):
    """Izlistaj šta je unutar zip fajla bez da ga ekstrahuje.

    Ovaj dataset je napravljen kao arhive unutar arhiva, 
    Tako da je korisno da se vidi koliko je diskovnog prostora potrebno i kakav je unutrašnji raspored.
    """
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            total = sum(i.file_size for i in z.infolist())
            exts = collections.Counter(os.path.splitext(n)[1].lower()
                                       for n in names if not n.endswith("/"))
            print(f"{indent}{len(names)} entries, {human(total)} uncompressed")
            print(f"{indent}types: "
                  + ", ".join(f"{e or '(none)'} x{c}" for e, c in exts.most_common(8)))
            for n in names[:limit]:
                print(f"{indent}  {n}")
            if len(names) > limit:
                print(f"{indent}  ... {len(names) - limit} more")
    except Exception as e:
        print(f"{indent}(could not read archive: {e})")


def probe_video(path, indent="    "):
    """Reportovanje trajanja, fps i rezolucije ako je ffprobe dostupan."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
             "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        info = json.loads(out.stdout)
        s = (info.get("streams") or [{}])[0]
        dur = float(info.get("format", {}).get("duration", 0) or 0)
        num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0
        print(f"{indent}{s.get('width')}x{s.get('height')}  "
              f"{fps:.2f} fps  {dur:.1f} s  "
              f"~{s.get('nb_frames') or int(dur * fps)} frames")
        return dur, fps
    except FileNotFoundError:
        print(f"{indent}(ffprobe not installed — brew install ffmpeg)")
    except Exception as e:
        print(f"{indent}(probe failed: {e})")
    return 0, 0


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python 06_inspect_marvel.py /path/to/extracted/marvel")
    root = sys.argv[1]
    if not os.path.isdir(root):
        sys.exit(f"Not a directory: {root}")

    by_ext = collections.Counter()
    size_by_ext = collections.Counter()
    files = []

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.startswith("."):
                continue
            p = os.path.join(dirpath, fn)
            ext = os.path.splitext(fn)[1].lower()
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            by_ext[ext] += 1
            size_by_ext[ext] += size
            files.append((p, ext, size))

    print(f"=== {root} ===")
    print(f"{len(files)} files, {human(sum(s for _, _, s in files))} total\n")

    print("--- by extension ---")
    for ext, count in by_ext.most_common():
        print(f"  {ext or '(none)':<10} {count:>6} files   {human(size_by_ext[ext])}")

    print("\n--- directory tree (2 levels) ---")
    base_depth = root.rstrip("/").count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.count(os.sep) - base_depth
        if depth > 2:
            dirnames[:] = []
            continue
        print(f"  {'  ' * depth}{os.path.basename(dirpath) or root}/  "
              f"({len(filenames)} files)")

    # --- annotation files ------------------------------
    ann = [f for f in files if f[1] in TEXTLIKE]
    print(f"\n--- annotation-like files ({len(ann)}) ---")
    for p, ext, size in sorted(ann, key=lambda x: -x[2])[:10]:
        print(f"\n  {os.path.relpath(p, root)}  ({human(size)})")
        if ext == ".json":
            describe_json(p)
        else:
            describe_text(p)

    # --- archives ----------------------------------------------------------
    # Provera bez ekstrakcije, jer dataset je napravljen kao arhive unutar arhiva.
    arcs = [f for f in files if f[1] in ARCHIVE]
    if arcs:
        print(f"\n--- archives ({len(arcs)}), contents listed without extracting ---")
        for p, _, size in sorted(arcs, key=lambda x: -x[2]):
            print(f"\n  {os.path.relpath(p, root)}  ({human(size)} compressed)")
            describe_zip(p)

    # --- video -------------------------------------------------------------
    vids = [f for f in files if f[1] in VIDEO]
    if vids:
        print(f"\n--- video ({len(vids)} files) ---")
        total_dur = 0
        for p, _, size in sorted(vids, key=lambda x: -x[2])[:5]:
            print(f"  {os.path.relpath(p, root)}  ({human(size)})")
            dur, _ = probe_video(p)
            total_dur += dur
        if total_dur:
            print(f"\n  Sampled duration: {total_dur / 60:.1f} min across "
                  f"{min(5, len(vids))} files")
            print("  At 25 fps that is tens of thousands of frames. Do NOT run")
            print("  every frame — consecutive frames are nearly identical.")
            print("  Subsample (every 10th or 25th) for the main evaluation and")
            print("  reserve dense runs for short clips where you are measuring")
            print("  temporal stability specifically.")

    imgs = [f for f in files if f[1] in IMAGE]
    auds = [f for f in files if f[1] in AUDIO]
    if imgs:
        print(f"\n--- images: {len(imgs)} files, {human(sum(s for _,_,s in imgs))} ---")
    if auds:
        print(f"\n--- audio: {len(auds)} files (not needed for this project) ---")

    print("\n=== what to determine from the above ===")
    print("  1. Are boxes per-frame, or per time interval?")
    print("  2. Box format: [x,y,w,h] or [x1,y1,x2,y2]? Pixels or normalised?")
    print("  3. How do annotations reference frames — index, timestamp, filename?")
    print("  4. Class labels for bicycle / pedestrian / motorcycle.")
    print("  5. Confirm there are no segmentation masks (expected: there are none).")


if __name__ == "__main__":
    main()
