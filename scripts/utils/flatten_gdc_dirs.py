#!/usr/bin/env python3
"""Verify + flatten gdc-client <file_id>/ subfolders into the image directory.

Usage:
  python scripts/utils/flatten_gdc_dirs.py [--dir image] [--dry-run] [--md5]
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from svs_manifest import gdc_fetch, md5sum, human  # noqa: E402

UUID_LEN = 36


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="image directory")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--md5", action="store_true", help="also verify md5sums")
    ap.add_argument("--keep-dirs", action="store_true", help="keep the uuid folders")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    image_dir = args.dir or os.path.join(root, "image")

    # collect <file_id>/<name>.svs
    jobs = []
    for entry in sorted(os.listdir(image_dir)):
        sub = os.path.join(image_dir, entry)
        if not os.path.isdir(sub):
            continue
        svss = [f for f in sorted(os.listdir(sub)) if f.lower().endswith(".svs")]
        partials = [f for f in sorted(os.listdir(sub)) if f.lower().endswith(".partial")]
        for name in svss:
            jobs.append((entry, name, os.path.join(sub, name)))
        if partials:
            print(f"SKIP {entry}: interrupted download ({', '.join(partials)})")

    if not jobs:
        print("Nothing to flatten.")
        return 0

    meta = gdc_fetch([fid for fid, _, _ in jobs])

    ok, bad = [], []
    for fid, name, path in jobs:
        local = os.path.getsize(path)
        rec = meta.get(fid, {})
        expected = rec.get("size", 0)
        status = "OK"
        if expected and local != expected:
            status = f"SIZE MISMATCH (local {human(local)} vs GDC {human(expected)})"
            bad.append((fid, name, status))
        elif args.md5 and rec.get("md5"):
            if md5sum(path) != rec["md5"]:
                status = "MD5 MISMATCH"
                bad.append((fid, name, status))
        if status == "OK":
            ok.append((fid, name, path))
        print(f"{status:12} {name}  ({human(local)})")

    print(f"\n{len(ok)} file(s) complete, {len(bad)} problem file(s)")
    if bad:
        print("Problem files were left in place.")
        return 1

    if args.dry_run:
        print("Dry run - nothing moved.")
        return 0

    moved = 0
    for fid, name, path in ok:
        dest = os.path.join(image_dir, name)
        if os.path.exists(dest):
            print(f"SKIP {name}: already present in {image_dir}")
            continue
        os.rename(path, dest)
        moved += 1
        print(f"MOVED {name}")

    if not args.keep_dirs:
        for fid, name, path in ok:
            sub = os.path.join(image_dir, fid)
            if os.path.isdir(sub):
                try:
                    shutil.rmtree(sub)          # uuid folder + gdc-client logs
                except OSError as exc:
                    print(f"Note: could not remove {sub} ({exc})")
    print(f"\nFlattened {moved} slide(s) into {image_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
