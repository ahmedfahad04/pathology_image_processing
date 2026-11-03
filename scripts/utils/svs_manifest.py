#!/usr/bin/env python3
"""
Resolve slide names -> gdc-client manifest, skipping files already on disk.

Purpose:
  Companion helper for ``scripts/utils/download_svs.sh``.  It turns whatever
  the user asked for (slide file names, TCGA sample IDs, TCGA patient IDs, a
  cBioPortal mutation list, ...) into an ordered list of GDC slide files and
  then decides which of them still need to be downloaded.

  1. Reads the slide TSV (``output/misc/tcga_blca_slides.tsv``) and indexes
     every slide by file name, stem, sample ID and patient ID.
  2. Resolves the requested tokens (args / ``--list`` files / stdin) into
     slides.  A token can be:
       * ``TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-....svs``  (exact slide file name)
       * ``TCGA-FJ-A871-01``                            (sample ID -> all slides)
       * ``TCGA-FJ-A871``                               (patient ID -> all slides)
       * ``TCGA-FJ-A871-01Z``                           (slide-prefix -> matching slides)
     Unknown tokens are looked up in the GDC API as a last resort.
  3. Applies an optional 1-based inclusive range (``--range 10:20``) to the
     *resolved* list, so a long cohort can be downloaded in batches.
  4. Checks every slide against the local output directory:
       * missing                  -> download
       * present, size matches     -> skip (already downloaded)
       * present, size too small   -> partial file, re-download
       * present, size differs     -> corrupt, re-download (unless --no-size-check)
       * ``--verify``               -> md5 compared against the GDC md5sum
  5. Writes a gdc-client manifest (``id  filename  md5  size  state``).

Barcode parsing (https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/):
  TCGA-FJ-A871-01Z-00-DX5.8F79D0A8-5DE6-4159-AA77-61DACB21E867.svs
    -> patient_id = TCGA-FJ-A871
    -> sample_id  = TCGA-FJ-A871-01

Usage:
  # every slide of the samples listed in the cBioPortal mutation file
  python scripts/utils/svs_manifest.py --list output/paper/mutations_blca_tcga_pancancer_atlas_cbioportal.txt \
      --outdir image --manifest /tmp/manifest.txt

  # only slides 10..20 of that list (1-based, inclusive)
  python scripts/utils/svs_manifest.py --list output/paper/mutations_blca_tcga_pancancer_atlas_cbioportal.txt \
      --range 10:20 --outdir image --manifest /tmp/manifest.txt

  # what would be downloaded? (no manifest needed)
  python scripts/utils/svs_manifest.py --list list.txt --outdir image --dry-run
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

GDC_API = "https://api.gdc.cancer.gov/files"
BATCH_SIZE = 500

# ---------------------------------------------------------------- utilities


def die(msg: str, code: int = 1) -> "None":
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg: str) -> None:
    print(f"Warning: {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(msg, file=sys.stderr)


def pick(row: Dict[str, str], *names: str) -> Optional[str]:
    """Return the first non-empty value of ``row`` whose key matches ``names``.

    Matching is case-insensitive and also accepts a key that *ends with* one
    of the names (the GDC TSV uses ``cases.0.submitter_id``).
    """
    lowered = {k.lower(): v for k, v in row.items() if k}
    for name in names:
        if name in row and row[name]:
            return row[name].strip()
        for key, value in lowered.items():
            if not value:
                continue
            if key == name.lower() or key.endswith("." + name.lower()) or key.endswith("_" + name.lower()):
                return value.strip()
    return None


def md5sum(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def human(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}" if unit != "B" else f"{int(num)}B"
        num /= 1024.0
    return f"{num:.1f}PB"


# ---------------------------------------------------------------- GDC API


def gdc_fetch(ids: Sequence[str], api: str = GDC_API, timeout: int = 120) -> Dict[str, Dict]:
    """Fetch metadata for ``ids`` from the GDC API (batched POST)."""
    meta: Dict[str, Dict] = {}
    ids = [i for i in dict.fromkeys(ids) if i]
    for start in range(0, len(ids), BATCH_SIZE):
        chunk = ids[start:start + BATCH_SIZE]
        payload = {
            "filters": {"op": "in", "content": {"field": "file_id", "value": chunk}},
            "fields": "file_id,file_name,md5sum,file_size,state",
            "size": str(len(chunk)),
            "format": "TSV",
        }
        text = _gdc_post(payload, api, timeout)
        if not text:
            continue
        lines = text.strip().splitlines()
        if not lines:
            continue
        header = lines[0].split("\t")
        for line in lines[1:]:
            if not line.strip():
                continue
            record = dict(zip(header, line.split("\t")))
            file_id = record.get("file_id") or record.get("id")
            if not file_id:
                continue
            try:
                size = int(record.get("file_size") or 0)
            except ValueError:
                size = 0
            meta[file_id] = {
                "file_id": file_id,
                "file_name": record.get("file_name", ""),
                "md5": record.get("md5sum", ""),
                "size": size,
                "state": record.get("state", ""),
            }
    return meta


def _gdc_post(payload: Dict, api: str, timeout: int) -> str:
    """POST to the GDC API; urllib first, curl as a fallback (proxy setups)."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        api, data=body, headers={"Content-Type": "application/json", "Accept": "text/plain"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - fall back to curl
        try:
            proc = subprocess.run(
                ["curl", "-sS", "-m", str(timeout), "-X", "POST", api,
                 "-H", "Content-Type: application/json", "--data-binary", "@-"],
                input=body, capture_output=True, timeout=timeout + 30,
            )
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        warn(f"GDC API request failed ({exc})")
        return ""


def gdc_lookup_by_name(file_names: Sequence[str], api: str = GDC_API, timeout: int = 120) -> Dict[str, Dict]:
    """Look up slides by their exact file name (used for tokens missing from the TSV)."""
    meta: Dict[str, Dict] = {}
    names = [n for n in dict.fromkeys(file_names) if n]
    for start in range(0, len(names), BATCH_SIZE):
        chunk = names[start:start + BATCH_SIZE]
        payload = {
            "filters": {"op": "in", "content": {"field": "file_name", "value": chunk}},
            "fields": "file_id,file_name,md5sum,file_size,state",
            "size": str(len(chunk)),
            "format": "TSV",
        }
        text = _gdc_post(payload, api, timeout)
        if not text:
            continue
        lines = text.strip().splitlines()
        if not lines:
            continue
        header = lines[0].split("\t")
        for line in lines[1:]:
            record = dict(zip(header, line.split("\t")))
            file_id = record.get("file_id") or record.get("id")
            if not file_id:
                continue
            try:
                size = int(record.get("file_size") or 0)
            except ValueError:
                size = 0
            meta[record.get("file_name", "")] = {
                "file_id": file_id,
                "file_name": record.get("file_name", ""),
                "md5": record.get("md5sum", ""),
                "size": size,
                "state": record.get("state", ""),
            }
    return meta


# ---------------------------------------------------------------- slide index


class SlideIndex:
    """Index of the slides listed in the TSV (file name / stem / sample / patient)."""

    def __init__(self, tsv_path: str) -> None:
        self.rows: List[Dict[str, str]] = []
        self.by_name: Dict[str, Dict] = {}
        self.by_stem: Dict[str, Dict] = {}
        self.by_sample: Dict[str, List[Dict]] = {}
        self.by_patient: Dict[str, List[Dict]] = {}
        self._load(tsv_path)

    def _load(self, tsv_path: str) -> None:
        with open(tsv_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header: Optional[List[str]] = None
            for raw in reader:
                if not raw:
                    continue
                if header is None:
                    header = [h.strip() for h in raw]
                    continue
                row = dict(zip(header, raw))
                file_name = pick(row, "file_name", "filename", "file")
                if not file_name:
                    continue
                file_id = pick(row, "file_id", "id", "uuid") or ""
                patient = pick(row, "submitter_id", "patient_id", "case_id", "cases.0.submitter_id") or ""
                # "cases.0.case_id" is a UUID, not a TCGA barcode -> ignore it
                if not patient.upper().startswith("TCGA-"):
                    patient = ""
                entry = {
                    "file_name": file_name,
                    "file_id": file_id,
                    "patient_id": patient,
                    "sample_id": f"{patient}-01" if patient else "",
                }
                self.rows.append(entry)
                self.by_name[file_name] = entry
                self.by_stem[os.path.splitext(file_name)[0]] = entry
                if entry["sample_id"]:
                    self.by_sample.setdefault(entry["sample_id"], []).append(entry)
                if patient:
                    self.by_patient.setdefault(patient, []).append(entry)

    def lookup(self, token: str) -> List[Dict]:
        """Return the slides matching ``token`` (may be empty)."""
        token = token.strip()
        if not token:
            return []

        entry = self.by_name.get(token)
        if entry:
            return [entry]
        entry = self.by_stem.get(token)
        if entry:
            return [entry]

        # sample id (TCGA-..-..-01) or patient id (TCGA-..-..)
        if token in self.by_sample:
            return list(self.by_sample[token])
        if token in self.by_patient:
            return list(self.by_patient[token])

        # slide prefix, e.g. TCGA-FJ-A871-01Z  ->  TCGA-FJ-A871-01Z-00-DX5....svs
        if token.upper().startswith("TCGA-"):
            upper = token.upper()
            matches = [
                row for row in self.rows
                if row["file_name"].upper().startswith(upper + "-")
                or os.path.splitext(row["file_name"])[0].upper().startswith(upper)
            ]
            if matches:
                return matches
        return []


# ---------------------------------------------------------------- list parsing


FILE_NAME_COLUMNS = ("file_name", "filename", "file", "slide", "image")
SAMPLE_COLUMNS = ("sample_id", "sample", "sampleid", "submitted_sample_id")
PATIENT_COLUMNS = ("patient_id", "patient", "submitter_id", "case_id")

# column groups -> kind of value they hold
COLUMN_GROUPS = (
    (FILE_NAME_COLUMNS, "file"),
    (SAMPLE_COLUMNS, "sample"),
    (PATIENT_COLUMNS, "patient"),
)


def read_tokens(list_path: str) -> List[str]:
    """Read a list file and return the tokens to resolve.

    Supports:
      * a directory        - every ``*.svs`` file inside it
      * plain text         - one slide name / sample id per line
      * TSV / CSV          - a column named ``file_name`` / ``sample_id`` / ``patient_id``
      * the cBioPortal mutation export (tab separated, ``SAMPLE_ID`` column)
    """
    tokens: List[str] = []
    if os.path.isdir(list_path):
        for name in sorted(os.listdir(list_path)):
            if name.lower().endswith(".svs"):
                tokens.append(name)
        return tokens
    with open(list_path, newline="", encoding="utf-8", errors="replace") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        delimiter = "\t" if sample.count("\t") >= 1 else ("," if sample.count(",") >= 1 else None)

        if delimiter is None:
            for line in fh:
                tokens.extend(_split_line(line))
            return tokens

        reader = csv.reader(fh, delimiter=delimiter)
        header: Optional[List[str]] = None
        for row in reader:
            if not row or all(not c.strip() for c in row):
                continue
            if header is None:
                candidate_header = [c.strip() for c in row]
                column = _pick_column(candidate_header)
                if column is None:
                    # no header we understand -> treat every field as a token
                    header = []
                    tokens.extend(_split_row(row))
                else:
                    header = candidate_header
                continue
            column = _pick_column(header)
            if column is None:
                tokens.extend(_split_row(row))
                continue
            index, _kind = column
            value = row[index].strip() if index < len(row) else ""
            if value:
                tokens.append(value)
    return tokens


def _pick_column(header: List[str]) -> Optional[Tuple[int, str]]:
    """Return ``(index, kind)`` of the first column matching a known group."""
    for group, kind in COLUMN_GROUPS:
        for index, name in enumerate(header):
            low = name.lower()
            for candidate in group:
                if low == candidate or low.endswith("." + candidate) or low.endswith("_" + candidate):
                    return index, kind
    return None


def _split_line(line: str) -> List[str]:
    parts = re.split(r"[\s,;]+", line.strip().strip('"'))
    return [p for p in parts if p]


def _split_row(row: List[str]) -> List[str]:
    out: List[str] = []
    for cell in row:
        out.extend(_split_line(cell))
    return out


# ---------------------------------------------------------------- range


def parse_range(spec: Optional[str], start: Optional[int], end: Optional[int], count: Optional[int],
                total: int) -> Tuple[int, int]:
    """Return a 1-based inclusive ``(first, last)`` slice of ``total`` items."""
    first, last = 1, total

    if spec:
        text = spec.strip()
        if ":" in text:
            left, _, right = text.partition(":")
        elif "-" in text:
            left, _, right = text.partition("-")
        else:
            left, right = text, ""
        first = int(left) if left.strip() else 1
        last = int(right) if right.strip() else total
    if start is not None:
        first = start
    if end is not None:
        last = end
    if count is not None:
        last = first + count - 1

    if first < 1:
        first = 1
    if last < first:
        last = first
    if last > total:
        last = total
    return first, last


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve slide names to a gdc-client manifest, skipping files already on disk.",
    )
    parser.add_argument("--tsv", default=None,
                        help="slide TSV (default: <project>/output/misc/tcga_blca_slides.tsv)")
    parser.add_argument("--outdir", default=None,
                        help="directory that holds the downloaded slides (default: <project>/image)")
    parser.add_argument("--list", action="append", default=[], metavar="FILE",
                        help="list file with slide names / sample ids (repeatable)")
    parser.add_argument("--token", action="append", default=[], metavar="TOKEN",
                        help="single slide name or sample id (repeatable)")
    parser.add_argument("--range", dest="range_spec", default=None, metavar="START:END",
                        help="1-based inclusive range over the resolved list (START-END also accepted)")
    parser.add_argument("--start", type=int, default=None, help="1-based start index")
    parser.add_argument("--end", type=int, default=None, help="1-based end index")
    parser.add_argument("--count", type=int, default=None, help="number of slides starting at --start")
    parser.add_argument("--pattern", default=None,
                        help="only keep slides whose file name matches this glob (e.g. '*-01Z-00-DX1*')")
    parser.add_argument("--manifest", default=None, help="write the gdc-client manifest here")
    parser.add_argument("--save-list", default=None,
                        help="write the resolved slide list (one file name per line) here")
    parser.add_argument("--save-map", default=None,
                        help="write the resolved slides as 'file_id<TAB>file_name' here")
    parser.add_argument("--dry-run", action="store_true", help="only report what would be downloaded")
    parser.add_argument("--verify", action="store_true", help="md5-verify slides already on disk")
    parser.add_argument("--force", action="store_true", help="re-download even if the slide is present")
    parser.add_argument("--no-size-check", action="store_true",
                        help="treat any existing file as complete (skip the size comparison)")
    parser.add_argument("--api", default=GDC_API, help="GDC files API endpoint")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    tsv = args.tsv or os.path.join(project_root, "output", "misc", "tcga_blca_slides.tsv")
    outdir = args.outdir or os.path.join(project_root, "image")

    if not os.path.isfile(tsv):
        die(f"slide TSV not found: {tsv}")

    index = SlideIndex(tsv)
    info(f"Loaded {len(index.rows)} slides from {tsv}")

    # ---- collect the requested tokens -------------------------------------
    tokens: List[str] = list(args.token)
    for list_path in args.list:
        if not os.path.exists(list_path):
            die(f"list file not found: {list_path}")
        file_tokens = read_tokens(list_path)
        info(f"  {list_path}: {len(file_tokens)} entr(ies)")
        tokens.extend(file_tokens)
    if not tokens and not sys.stdin.isatty():
        stdin_tokens = _split_line(sys.stdin.read())
        tokens.extend(stdin_tokens)
        if tokens:
            info(f"  stdin: {len(tokens)} entr(ies)")

    if not tokens:
        die("nothing to download: pass slide names as arguments, --list FILE, or pipe a list on stdin")

    # ---- resolve ----------------------------------------------------------
    resolved: List[Dict] = []
    seen: set = set()
    unresolved: List[str] = []
    unknown_names: List[str] = []

    def _add(entry: Dict) -> None:
        if entry["file_name"] in seen:
            return
        seen.add(entry["file_name"])
        resolved.append(entry)

    for token in tokens:
        matches = index.lookup(token)
        if matches:
            for entry in matches:
                _add(entry)
            continue
        if token.lower().endswith(".svs"):
            unknown_names.append(token)   # resolved through the GDC API below
        else:
            unresolved.append(token)

    if unknown_names:
        info(f"Resolving {len(unknown_names)} name(s) not present in the TSV via the GDC API ...")
        found = gdc_lookup_by_name(unknown_names, api=args.api)
        for name in unknown_names:
            record = found.get(name)
            if record:
                _add({"file_name": record["file_name"], "file_id": record["file_id"],
                      "patient_id": "", "sample_id": ""})
            else:
                unresolved.append(name)

    if unresolved:
        for token in unresolved[:20]:
            warn(f"could not resolve '{token}' - skipped")
        if len(unresolved) > 20:
            warn(f"... and {len(unresolved) - 20} more unresolved entries")

    if args.pattern:
        before = len(resolved)
        resolved = [r for r in resolved if fnmatch.fnmatch(r["file_name"], args.pattern)]
        info(f"Pattern '{args.pattern}': kept {len(resolved)}/{before} slide(s)")

    if not resolved:
        die("no slides could be resolved from the given input")

    # ---- range ------------------------------------------------------------
    first, last = parse_range(args.range_spec, args.start, args.end, args.count, len(resolved))
    selected = resolved[first - 1:last] if last >= first else []
    if args.range_spec or args.start or args.end or args.count:
        info(f"Range {first}..{last} of {len(resolved)} resolved slide(s) -> {len(selected)} selected")
    if not selected:
        info("Nothing selected.")
        return 0

    # ---- metadata ---------------------------------------------------------
    file_ids = [r["file_id"] for r in selected if r.get("file_id")]
    missing_ids = [r for r in selected if not r.get("file_id")]
    meta = gdc_fetch(file_ids, api=args.api)
    for entry in missing_ids:
        record = gdc_lookup_by_name([entry["file_name"]], api=args.api).get(entry["file_name"])
        if record:
            entry["file_id"] = record["file_id"]
            meta[record["file_id"]] = record

    # ---- presence check ---------------------------------------------------
    os.makedirs(outdir, exist_ok=True)
    to_download: List[Dict] = []
    present = 0
    partial = 0
    forced_count = 0
    total_bytes = 0

    for entry in selected:
        file_name = entry["file_name"]
        file_id = entry.get("file_id", "")
        record = meta.get(file_id, {})
        state = record.get("state", "")
        if state and state != "released":
            warn(f"{file_name}: GDC state is '{state}' - skipped")
            continue

        path = os.path.join(outdir, file_name)
        expected = record.get("size", 0)
        reason = None

        if not os.path.isfile(path) and file_id:
            # gdc-client stores files as <outdir>/<file_id>/<file_name>; a run
            # that was killed before the flattening step leaves them there
            nested = os.path.join(outdir, file_id, file_name)
            if os.path.isfile(nested):
                path = nested

        if args.force:
            reason = "forced"
        elif not os.path.isfile(path):
            reason = "missing"
        else:
            local_size = os.path.getsize(path)
            if local_size == 0:
                reason = "empty file"
            elif not args.no_size_check and expected and local_size != expected:
                reason = ("partial" if local_size < expected else "size mismatch")
            elif args.verify and record.get("md5"):
                if md5sum(path) != record["md5"]:
                    reason = "md5 mismatch"

        if reason is None:
            present += 1
            continue

        if reason in ("partial", "size mismatch", "md5 mismatch", "empty file", "forced"):
            if reason == "forced":
                forced_count += 1
            else:
                partial += 1
            local = os.path.getsize(path) if os.path.isfile(path) else 0
            info(f"re-downloading {file_name} ({reason}: {human(local)} vs {human(expected)})")
            if not args.dry_run:
                # gdc-client skips files that already match the manifest md5,
                # so the local copy has to go before a forced re-download
                try:
                    os.remove(path)
                except OSError:
                    pass

        to_download.append({"file_name": file_name, "file_id": file_id, "meta": record})
        total_bytes += expected

    info("")
    info(f"Resolved : {len(selected)} slide(s)")
    info(f"Present  : {present} (already downloaded, skipped)")
    info(f"To fetch  : {len(to_download)} slide(s)  ~{human(total_bytes)}")
    if partial:
        info(f"Replaced : {partial} incomplete/corrupt file(s)")
    if forced_count:
        info(f"Forced   : {forced_count} file(s) re-downloaded on request")

    if args.save_list:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_list)), exist_ok=True)
        with open(args.save_list, "w", encoding="utf-8") as fh:
            for entry in selected:
                fh.write(entry["file_name"] + "\n")
        info(f"Resolved list written to {args.save_list}")

    if args.save_map:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_map)), exist_ok=True)
        with open(args.save_map, "w", encoding="utf-8") as fh:
            for entry in selected:
                fh.write(f"{entry.get('file_id', '')}\t{entry['file_name']}\n")
        info(f"Resolved map written to {args.save_map}")

    if args.dry_run:
        info("")
        info("Dry run - nothing was downloaded. Files that would be fetched:")
        for item in to_download:
            info(f"  {item['file_name']}")
        return 0

    # ---- manifest ---------------------------------------------------------
    lines = ["id\tfilename\tmd5\tsize\tstate"]
    for item in to_download:
        record = item["meta"]
        lines.append("\t".join([
            item["file_id"],
            item["file_name"],
            record.get("md5", ""),
            str(record.get("size", 0)),
            record.get("state", "released"),
        ]))
    manifest_text = "\n".join(lines) + "\n"

    if not to_download:
        info("Nothing to do: every requested slide is already present.")
    if args.manifest:
        os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
        with open(args.manifest, "w", encoding="utf-8") as fh:
            fh.write(manifest_text)
        info(f"Manifest written to {args.manifest}")
    else:
        sys.stdout.write(manifest_text)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
