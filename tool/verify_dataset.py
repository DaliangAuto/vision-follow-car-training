#!/usr/bin/env python3
"""
Verify dataset: check photo IDs vs CSV match (1:1, photos as reference) and sort CSV by frame_idx.

Usage:
  python verify_dataset.py [--path PATH] [--fix]
  python verify_dataset.py                    # Check dataset/0312
  python verify_dataset.py --path dataset/0312
  python verify_dataset.py --fix              # Sort CSV, remove rows without matching photos
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def parse_frame_id(path_or_stem):
    """Extract frame ID from path (e.g. frames/000302.jpg or 000302) -> 302."""
    s = str(path_or_stem)
    m = re.search(r"(\d+)\.(?:jpg|jpeg|png)$", s, re.I)
    if m:
        return int(m.group(1))
    if s.isdigit():
        return int(s)
    return None


def get_frames_dir(dir_path):
    """Return frames subdir if exists, else the dir itself."""
    frames_sub = dir_path / "frames"
    return frames_sub if frames_sub.is_dir() else dir_path


def get_photo_ids(dir_path):
    """Get set of frame IDs from photos in dir/frames/ or dir/."""
    d = get_frames_dir(dir_path)
    if not d.exists():
        return set()
    ids = set()
    for f in d.iterdir():
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            fid = parse_frame_id(f.stem)
            if fid is not None:
                ids.add(fid)
    return ids


def get_csv_rows_with_ids(csv_path):
    """Read CSV, return (headers, list of (frame_id, row_dict))."""
    if not csv_path.exists():
        return None, []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if not headers:
            return None, []
        rows_with_id = []
        for r in reader:
            fid = None
            if "frame_idx" in r and str(r.get("frame_idx", "")).strip():
                try:
                    fid = int(r["frame_idx"])
                except (ValueError, TypeError):
                    pass
            if fid is None and "image_path" in r:
                fid = parse_frame_id(r["image_path"])
            if fid is not None:
                rows_with_id.append((fid, dict(r)))
        return headers, rows_with_id


def verify_and_fix_dir(dir_path, fix=False):
    """
    Verify photos vs CSV 1:1 match (photos as reference).
    If fix: sort CSV by frame_idx, remove rows without matching photos, dedupe by frame_id.
    Returns (ok, messages).
    """
    csv_path = dir_path / "controls.csv"
    if not csv_path.exists():
        return True, [f"  [SKIP] no controls.csv"]

    photo_ids = get_photo_ids(dir_path)
    headers, csv_rows = get_csv_rows_with_ids(csv_path)

    if not headers:
        return False, [f"  [ERROR] empty or invalid CSV"]

    # Build: frame_id -> [rows]
    csv_by_id = {}
    for fid, row in csv_rows:
        csv_by_id.setdefault(fid, []).append(row)

    csv_ids = set(csv_by_id.keys())
    in_csv_not_in_photos = csv_ids - photo_ids
    in_photos_not_in_csv = photo_ids - csv_ids
    dupe_ids = {fid for fid, rows in csv_by_id.items() if len(rows) > 1}

    messages = []
    ok = True

    messages.append(f"  photos: {len(photo_ids)}, CSV rows: {len(csv_rows)}")

    if in_photos_not_in_csv:
        ok = False
        missing = sorted(in_photos_not_in_csv)
        messages.append(f"  [MISMATCH] photos without CSV row ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}")

    if in_csv_not_in_photos:
        extra = sorted(in_csv_not_in_photos)
        messages.append(f"  [MISMATCH] CSV rows without photo ({len(extra)}): {extra[:20]}{'...' if len(extra) > 20 else ''}")
        if fix:
            for fid in in_csv_not_in_photos:
                del csv_by_id[fid]

    if dupe_ids:
        messages.append(f"  [DUPES] duplicate frame_idx in CSV ({len(dupe_ids)}): {sorted(dupe_ids)[:10]}{'...' if len(dupe_ids) > 10 else ''}")
        if fix:
            for fid in dupe_ids:
                csv_by_id[fid] = [csv_by_id[fid][0]]

    if not in_csv_not_in_photos and not in_photos_not_in_csv and not dupe_ids:
        messages.append("  OK (1:1 match)")

    if fix and (in_csv_not_in_photos or dupe_ids or True):
        # Sort by frame_idx and write back
        sorted_ids = sorted(csv_by_id.keys())
        new_rows = []
        for fid in sorted_ids:
            new_rows.append(csv_by_id[fid][0])
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            w.writerows(new_rows)
        messages.append(f"  [FIXED] CSV sorted by frame_idx, {len(new_rows)} rows written")

    return ok, messages


def main():
    parser = argparse.ArgumentParser(description="Verify dataset: photos vs CSV 1:1 match, sort CSV by frame_idx")
    parser.add_argument("--path", type=str, default=None, help="Dataset root (default: dataset/0312)")
    parser.add_argument("--fix", action="store_true", help="Fix CSV: sort by frame_idx, remove rows without photos")
    args = parser.parse_args()

    proj = Path(__file__).resolve().parent.parent
    base = Path(args.path) if args.path else proj / "dataset" / "0312"
    base = base if base.is_absolute() else proj / base

    if not base.exists():
        print(f"[ERROR] Path not found: {base}")
        sys.exit(1)

    # Subdirs that typically have frames + controls.csv
    subdirs = ["main1", "main2", "no_target"]
    dirs_to_check = []
    for name in subdirs:
        d = base / name
        if d.is_dir() and (d / "controls.csv").exists():
            dirs_to_check.append((name, d))

    if not dirs_to_check:
        # Fallback: check base itself if it has controls.csv
        if (base / "controls.csv").exists():
            dirs_to_check = [("<root>", base)]
        else:
            print(f"[ERROR] No subdirs with controls.csv found under {base}")
            print("  Expected: main1, main2, no_target with controls.csv")
            sys.exit(1)

    print(f"Verifying: {base}")
    if args.fix:
        print("Mode: --fix (will sort CSV and remove rows without matching photos)")
    print()

    all_ok = True
    for name, dir_path in dirs_to_check:
        print(f"=== {name} ===")
        ok, msgs = verify_and_fix_dir(dir_path, fix=args.fix)
        for m in msgs:
            print(m)
        if not ok:
            all_ok = False
        print()

    if all_ok:
        print("All directories OK.")
    else:
        print("Some directories have mismatches. Use --fix to remove extra CSV rows and sort.")
        sys.exit(1)


if __name__ == "__main__":
    main()
