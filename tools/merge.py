#!/usr/bin/env python3
"""Validate every evidence row and fold it into one dataset.

Reads data/raw/*.jsonl (harvested per family) and data/staging/*.jsonl (curated by
tools/extract.py), checks each row against docs/schema.md, drops exact duplicates,
and writes:

    data/prefill.jsonl    the merged dataset, sorted
    data/summary.json     counts by family, dataset, store, provenance

Usage: tools/merge.py [--check]
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

STORES = {
    "vram-only", "ram-all", "nvme-single", "nvme-raid0", "nvme-capped",
    "unified-memory", "usb-or-oculink", "disk-engram", "unknown",
}
PROVENANCE = {"result-json", "registry-run", "sweep-json", "transcribed"}
REQUIRED = {
    "id": str, "family": str, "dataset": str, "model": str, "gpu": str,
    "store": str, "store_detail": str, "offload": bool, "protocol": str,
    "provenance": str, "date": str,
}


def check_row(row, path, lineno):
    """Return a list of problems with one row."""
    bad = []
    for field, typ in REQUIRED.items():
        if field not in row:
            bad.append("missing %s" % field)
        elif not isinstance(row[field], typ):
            bad.append("%s is %s, want %s" % (field, type(row[field]).__name__,
                                              typ.__name__))
    src = row.get("source") or {}
    if not isinstance(src, dict):
        bad.append("source is not an object")
        src = {}
    for field in ("repo", "commit", "path"):
        if not src.get(field):
            bad.append("missing source.%s" % field)
    commit = src.get("commit") or ""
    if commit and not re.fullmatch(r"[0-9a-f]{40}", commit):
        bad.append("source.commit is not a full sha: %r" % commit)
    if row.get("store") not in STORES:
        bad.append("store %r not in the vocabulary" % row.get("store"))
    if row.get("provenance") not in PROVENANCE:
        bad.append("provenance %r not in the vocabulary" % row.get("provenance"))
    if row.get("date_is_commit") not in (None, True, False):
        bad.append("date_is_commit must be a bool")
    if row.get("date") and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row["date"]):
        bad.append("date is not YYYY-MM-DD: %r" % row["date"])
    for field in ("prompt_tokens", "prefill_tok_s"):
        if not isinstance(row.get(field), (int, float)) or isinstance(row.get(field), bool):
            bad.append("%s missing or not numeric" % field)
    if isinstance(row.get("prefill_tok_s"), (int, float)) and row["prefill_tok_s"] <= 0:
        bad.append("prefill_tok_s must be positive")
    if isinstance(row.get("prompt_tokens"), (int, float)) and row["prompt_tokens"] <= 0:
        bad.append("prompt_tokens must be positive")
    if bad:
        return ["%s:%d %s: %s" % (os.path.basename(path), lineno,
                                  row.get("id", "<no id>"), "; ".join(bad))]
    return []


def load(pattern, problems):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    problems.append("%s:%d not JSON: %s" % (os.path.basename(path), i, e))
                    continue
                problems.extend(check_row(row, path, i))
                row.setdefault("_file", os.path.basename(path))
                rows.append(row)
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = ap.parse_args()

    problems = []
    rows = load(os.path.join(root, "data", "staging", "*.jsonl"), problems)
    rows += load(os.path.join(root, "data", "raw", "*.jsonl"), problems)
    if not rows:
        sys.exit("no rows found under data/staging or data/raw")

    # Duplicate ids: identical rows are fine (dedupe), conflicting ones are a problem.
    by_id = collections.defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)
    merged, dropped = [], 0
    for rid, group in by_id.items():
        first = group[0]
        for other in group[1:]:
            a = {k: v for k, v in first.items() if k != "_file"}
            b = {k: v for k, v in other.items() if k != "_file"}
            if a != b:
                problems.append("id %s appears in %s with different content"
                                % (rid, " and ".join(sorted({g["_file"] for g in group}))))
            dropped += 1
        merged.append(first)

    merged.sort(key=lambda r: (r["family"], r["dataset"], r["model"],
                               r["prompt_tokens"], r.get("date") or "", r["id"]))
    summary = {
        "rows": len(merged),
        "by_family": dict(collections.Counter(r["family"] for r in merged)),
        "by_dataset": dict(collections.Counter(r["dataset"] for r in merged)),
        "by_store": dict(collections.Counter(r["store"] for r in merged)),
        "by_provenance": dict(collections.Counter(r["provenance"] for r in merged)),
        "models": sorted({r["model"] for r in merged}),
        "hosts": sorted({r.get("host") or "unstated" for r in merged}),
        "duplicates_dropped": dropped,
    }

    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit("%d problem(s); nothing written" % len(problems))
    if args.check:
        print(json.dumps(summary, indent=1, sort_keys=True))
        return

    with open(os.path.join(root, "data", "prefill.jsonl"), "w") as f:
        for r in merged:
            r.pop("_file", None)
            f.write(json.dumps(r, sort_keys=True) + "\n")
    # Every (repo, commit, file) the dataset rests on, with how many rows cite it.
    cited = collections.Counter()
    for r in merged:
        src = r["source"]
        cited[(src["repo"], src["commit"], src["path"])] += 1
    with open(os.path.join(root, "data", "sources.json"), "w") as f:
        json.dump({
            "note": "Per-file sha256 of the machine-read subset is in "
                    "data/staging/sources.json; harvested rows carry the commit they "
                    "were read at.",
            "files": [
                {"repo": repo, "commit": commit, "path": path, "rows": n}
                for (repo, commit, path), n in sorted(cited.items())
            ],
        }, f, indent=1, sort_keys=True)
        f.write("\n")
    with open(os.path.join(root, "data", "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
