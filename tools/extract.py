#!/usr/bin/env python3
"""Extract the Lead's evidence rows (GLM-5.3-Flash offload + two length scans).

Writes staging files that tools/merge.py folds into data/prefill.jsonl:

    data/staging/glm-rows.jsonl        curated GLM-5.3-Flash run records + raw sweeps
    data/staging/dsv41-ctxscan.jsonl   DeepSeek-V4.1-Flash 8k-261k C1 scan (machine-read)
    data/staging/qwen-usb-scan.jsonl   Qwen3.8-Flash-Next over USB4 (transcribed)

Usage:
    tools/extract.py --sources ../            # dir holding the cloned repos
    tools/extract.py --sources ../ --check    # print, write nothing

Standard library only. Nothing is inferred that the source file does not state.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------- vocabulary

STORE_LABELS = {
    "nvme-single": (
        "1x Samsung 990 PRO 4 TB, Gen4 x4, in a motherboard M.2 slot, btrfs on LUKS. "
        "Measured read ceiling 7.45 GB/s."
    ),
    "nvme-raid0": (
        "4x Samsung 9100 PRO 1 TB, md RAID0, 512k chunk, xfs, on a PCIe 4.0 x16 "
        "quad-M.2 carrier card (bifurcated x4x4x4x4). Measured read ceiling 26.0 GB/s."
    ),
    "nvme-capped": (
        "The 4-drive RAID0 array with reads capped by docker --device-read-bps. The "
        "7 GB/s arms emulate one drive: same drives and queueing, one drive's bandwidth."
    ),
    "ram-all": (
        "All routed experts pinned in host RAM and streamed to VRAM over PCIe; no "
        "expert reads come from the NVMe store."
    ),
    "usb-or-oculink": (
        "Routed experts live on an M4 Mac mini's 256 GB SSD and are read over USB4."
    ),
}

GLM53 = {
    "family": "glm",
    "model": "GLM-5.3-Flash",
    "quant": "EXL3 3.05 bpw (turboderp), 42 MoE layers",
    "gpu": "1x RTX 3090 24 GB, PCIe 4.0 x16",
    "cpu": "AMD EPYC 7443P, 24 cores / 48 threads, 8-channel DDR4 503 GiB",
    "host": "omarchy",
    "offload": True,
}

GLM53_PROTOCOL = (
    "bench/sweep.py --template glm --prefill 8192 32768 --conc 1 2 4 --reps 3 "
    "--dec-reps 2: prefill = median of 3 fresh random-token prompts, tokens divided "
    "by time to first token"
)

DSV41 = {
    "family": "deepseek",
    "model": "DeepSeek-V4.1-Flash",
    "quant": "EXL3 3.0 bpw, 384 routed experts x 40 MoE layers",
    "gpu": "1x RTX 3090 24 GB, PCIe 4.0 x16",
    "cpu": "AMD EPYC 7443P, 24 cores / 48 threads, 256+ GiB DDR4",
    "host": "omarchy",
    "offload": True,
}

USB_MINI = {
    "family": "qwen",
    "model": "Qwen3.8-Flash-Next",
    "quant": "EXL3 3.05 bpw (turboderp)",
    "gpu": "1x RTX 3090 24 GB in a Chestnut eGPU",
    "cpu": "Apple M4 (Mac mini, 16 GB) + host cores",
    "host": "mini",
    "offload": True,
}

# ---------------------------------------------------------------- helpers


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(repo_dir):
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def parse_ram_cap(title, change, memory):
    """RAM cap of the container in GB, from the record itself."""
    if memory and isinstance(memory, dict) and memory.get("cgroup_max_gb"):
        return int(memory["cgroup_max_gb"])
    text = " ".join(x for x in (title, change) if x)
    m = re.search(r"--memory\s+(\d+)g", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*GB RAM cap", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*GB RAM", text)
    if m:
        return int(m.group(1))
    return None


def classify_store(run_id, title, change, recipe):
    if run_id in ("glm53-g066a-exact", "glm53-g067-allram"):
        return "ram-all"
    text = " ".join(x for x in (title, change, recipe) if x).lower()
    if "single 990 pro" in text or "glm53-single" in text:
        return "nvme-single"
    if "device-read-bps" in text or "read-bps" in text:
        return "nvme-capped"
    return "nvme-raid0"


def classify_mode(run_id, title, change):
    text = " ".join(x for x in (title, change) if x).lower()
    if "allram" in text or "all experts in ram" in text:
        return "all-RAM reference"
    if "b70" in text or "b70" in run_id:
        return "nvme + B70 expert tier"
    if "prefetch off" in text or "pfoff" in run_id:
        return "nvme, prefetch off"
    if run_id.startswith(("glm53-s1", "glm53-s2")):
        return "nvme-exact"
    return "nvme"


def io_block(phase):
    if not phase:
        return None
    eng = phase.get("engine") or {}
    out = {
        "seconds": phase.get("seconds"),
        "read_gbps": phase.get("read_gbps"),
        "iops": phase.get("device_iops"),
        "read_latency_ms": phase.get("device_read_latency_ms"),
        "pct_of_ceiling": phase.get("pct_of_ceiling"),
        "record_reads_per_token": eng.get("record_reads_per_token"),
        "stall_ms_per_token": eng.get("cpu_lane_nvme_wait_ms_per_token"),
    }
    return {k: v for k, v in out.items() if v is not None}


def read_json(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- GLM-5.3


def extract_glm53(src, commits, files):
    runs_dir = os.path.join(src, "moetier", "registry", "runs")
    rows = []
    for name in sorted(os.listdir(runs_dir)):
        if not name.startswith("glm53-") or not name.endswith(".json"):
            continue
        path = os.path.join(runs_dir, name)
        d = read_json(path)
        table = d.get("table") or []
        if not any(r.get("prefill_tok_s") for r in table):
            continue
        title, change = d.get("title"), d.get("change")
        store = classify_store(d["id"], title, change, d.get("recipe"))
        ram = parse_ram_cap(title, change, d.get("memory"))
        phases = (d.get("nvme_io") or {}).get("phases") or {}
        files.append(("sybil-solutions/moetier", commits["moetier"],
                      "registry/runs/" + name, sha256_file(path)))
        for entry in table:
            if not entry.get("prefill_tok_s") or entry.get("conc") != 1:
                continue
            ptoks = entry["prefill"]
            phase_key = "prefill_%dk" % (ptoks // 1024)
            ttft = entry.get("ttft_s")
            if isinstance(ttft, list):
                ttft = ttft[0] if ttft else None
            rows.append({
                "id": "glm53/%s@%s" % (d["id"], "8k" if ptoks == 8192 else "%gk" % (ptoks / 1024)),
                "dataset": "glm-flash-lite",
                **GLM53,
                "source": {
                    "repo": "sybil-solutions/moetier",
                    "commit": commits["moetier"],
                    "path": "registry/runs/" + name,
                    "run_id": d["id"],
                },
                "provenance": "registry-run",
                "date": d.get("measured"),
                "engine": d.get("image") or d.get("recipe"),
                "store": store,
                "store_detail": STORE_LABELS[store],
                "ram_cap_gb": ram,
                "mode": classify_mode(d["id"], title, change),
                "protocol": GLM53_PROTOCOL,
                "prompt_tokens": ptoks,
                "prefill_tok_s": entry["prefill_tok_s"],
                "ttft_s": ttft,
                "decode_tok_s": entry.get("decode"),
                "concurrency": 1,
                "io": io_block(phases.get(phase_key)),
                "run_title": title,
                "change": change,
                "notes": d.get("note"),
            })
    return rows


def extract_glm53_sweeps(src, commits, files):
    """Raw bench/sweep.py output: a second machine-read witness of the same runs."""
    mapping = {
        "G067": "glm53-g067-allram",
        "G066a": "glm53-g066a-exact",
        "N119-S2a-55g": "glm53-s2-exact-nvme-55g",
        "N119-S3a-55g": "glm53-s3a-cpu-nvme-55g",
        "N119-S3b2-55g": "glm53-s3b-cpu-nvme-batched-55g",
        "N129-pfon-55g": "glm53-55g-pfon-n129",
        "N129-s16-r48-16g": "glm53-16g-r48-screen",
        "N129-s16-r16-16g": "glm53-16g-r16-screen",
        "P003-nvme-image-smoke": "glm53-image-v4-nvme",
    }
    all_ram = ("G067", "G066a", "P001-image-smoke", "P002b-stream-fix-smoke")
    results = os.path.join(src, "glm-flash-lite", "results")
    rows = []
    for arm in sorted(os.listdir(results)):
        path = os.path.join(results, arm, "sweep.json")
        if not os.path.isfile(path):
            continue
        d = read_json(path)
        if not d.get("prefill"):
            continue
        files.append(("sybil-solutions/glm-flash-lite", commits["glm-flash-lite"],
                      "results/%s/sweep.json" % arm, sha256_file(path)))
        store = "ram-all" if arm in all_ram else "nvme-raid0"
        if arm not in all_ram and arm.startswith(("C052", "C056")):
            store = "unknown"
        for ptoks, stat in sorted(d["prefill"].items(), key=lambda kv: int(kv[0])):
            rows.append({
                "id": "glm53/sweep-%s@%s" % (arm, ptoks),
                "dataset": "glm-flash-lite",
                **GLM53,
                "source": {
                    "repo": "sybil-solutions/glm-flash-lite",
                    "commit": commits["glm-flash-lite"],
                    "path": "results/%s/sweep.json" % arm,
                    "run_id": arm,
                },
                "provenance": "sweep-json",
                "date": (d.get("started") or "")[:10] or None,
                "engine": d.get("config"),
                "store": store,
                "store_detail": STORE_LABELS.get(store, "The source does not state the store."),
                "ram_cap_gb": 55 if "55g" in arm else (16 if "16g" in arm else None),
                "mode": "all-RAM reference" if arm in all_ram else "nvme",
                "protocol": d.get("protocol") or GLM53_PROTOCOL,
                "prompt_tokens": int(ptoks),
                "prefill_tok_s": stat["median"],
                "ttft_s": None,
                "concurrency": 1,
                "io": None,
                "raw": {
                    "median": stat["median"], "min": stat["min"],
                    "max": stat["max"], "n": stat["n"],
                },
                "duplicate_of": mapping.get(arm),
                "notes": "raw sweep.json: repeats' min/max/n are in the raw block",
            })
    return rows


# ---------------------------------------------------------------- length scans


def extract_length_scans(src, commits, files):
    rows = []
    for fname in ("ctx_scan.json", "ctx_scan_261k.json"):
        path = os.path.join(src, "dsv41-flash-offload", "results", "D119", fname)
        if not os.path.isfile(path):
            continue
        d = read_json(path)
        files.append(("sybil-solutions/dsv41-flash-offload", commits["dsv41-flash-offload"],
                      "results/D119/" + fname, sha256_file(path)))
        for r in d.get("rows") or []:
            rows.append({
                "id": "dsv41/D119@%d" % r["prompt_tokens"],
                "dataset": "dsv41-flash-offload",
                **DSV41,
                "source": {
                    "repo": "sybil-solutions/dsv41-flash-offload",
                    "commit": commits["dsv41-flash-offload"],
                    "path": "results/D119/" + fname,
                    "run_id": "D119",
                },
                "provenance": "result-json",
                "date": (d.get("started") or "")[:10],
                "engine": "vLLM 0.13 + vllm_exl3 (D141 image)",
                "store": "ram-all",
                "store_detail": (
                    "All routed experts pinned in host DDR4 and DMA-staged into VRAM over "
                    "PCIe; the ~189 GiB Engram n-gram tables are memory-mapped on a single "
                    "Samsung 990 PRO 4 TB (dm-crypt), so that drive is read during prefill."
                ),
                "ram_cap_gb": None,
                "mode": "candidate (staged-DMA prefill)",
                "protocol": "bench/ctx_scan.py: prompt tokens / time to first token, C1, "
                            "greedy, natural EOS, one request per length",
                "prompt_tokens": r["prompt_tokens"],
                "prefill_tok_s": r["prefill_tok_s"],
                "ttft_s": r["ttft_s"],
                "concurrency": 1,
                "io": None,
                "notes": "single request per length; decode %.2f tok/s" % r["decode_tok_s"],
            })
    return rows, files


def extract_usb_scan(src, commits, files):
    """Qwen3.8-Flash-Next on a 3090 whose experts live on a Mac mini over USB4.

    Transcribed: this repo has raw logs, not result JSON.
    """
    path = os.path.join(src, "qwen38-flash-next-3090-usb-offload", "README.md")
    if not os.path.isfile(path):
        return [], files
    files.append(("sybil-solutions/qwen38-flash-next-3090-usb-offload",
                  commits["qwen38-flash-next-3090-usb-offload"], "README.md",
                  sha256_file(path)))
    text = open(path).read()
    base = {
        "dataset": "qwen38-flash-next-3090-usb-offload",
        **USB_MINI,
        "source": {
            "repo": "sybil-solutions/qwen38-flash-next-3090-usb-offload",
            "commit": commits["qwen38-flash-next-3090-usb-offload"],
            "path": "README.md",
            "run_id": "config C (Decode16)",
        },
        "provenance": "transcribed",
        "date": "2026-10-14",
        "engine": "exl3 + strata shim, config C",
        "store": "usb-or-oculink",
        "store_detail": STORE_LABELS["usb-or-oculink"],
        "ram_cap_gb": 16,
        "mode": "config C",
        "concurrency": 1,
    }
    section = "README.md, 'Results: config C' table"
    rows = []
    for m in re.finditer(r"^\|\s*([\d,]+)\s*\|\s*([\d.]+)\s+and\s+([\d.]+)\s*tok/s",
                         text, re.M):
        n = int(m.group(1).replace(",", ""))
        for tag, val in (("~a", m.group(2)), ("~b", m.group(3))):
            rows.append(dict(
                base,
                id="qwen38-usb/%d%s" % (n, tag),
                protocol="one request per length, C1; prefill tok/s only",
                prompt_tokens=n,
                prefill_tok_s=float(val),
                ttft_s=None,
                io=None,
                notes="%s; two runs of the same prompt shape with different random "
                      "first lines (no prefix cache)" % section,
            ))
    for m in re.finditer(
            r"\|\s*([\d,]+)\s*\([^)]*needle\)\s*\|\s*([\d.]+)\s*tok/s\s*\(TTFT\s*([\d.]+)\s*s\)",
            text):
        n = int(m.group(1).replace(",", ""))
        rows.append(dict(
            base,
            id="qwen38-usb/%d" % n,
            protocol="one request per length, C1; TTFT stated by the source",
            prompt_tokens=n,
            prefill_tok_s=float(m.group(2)),
            ttft_s=float(m.group(3)),
            io=None,
            notes="%s; needle-in-a-haystack row, TTFT stated" % section,
        ))
    return rows, files


# ---------------------------------------------------------------- main


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=os.path.join(here, "..", ".."),
                    help="directory holding moetier/, glm-flash-lite/, dsv41-flash-offload/, "
                         "qwen38-flash-next-3090-usb-offload/ clones")
    ap.add_argument("--out", default=os.path.join(here, "..", "data", "staging"))
    ap.add_argument("--check", action="store_true", help="print the rows, write nothing")
    args = ap.parse_args()

    src = os.path.abspath(args.sources)
    repos = ["moetier", "glm-flash-lite", "dsv41-flash-offload",
             "qwen38-flash-next-3090-usb-offload"]
    commits = {}
    for r in repos:
        p = os.path.join(src, r)
        if not os.path.isdir(p):
            sys.exit("missing clone: %s\nclone the four source repos next to this one, or "
                     "pass --sources" % p)
        commits[r] = git_commit(p)
        if not commits[r]:
            sys.exit("not a git checkout: %s" % p)

    files = []
    glm = extract_glm53(src, commits, files)
    sweeps = extract_glm53_sweeps(src, commits, files)
    scans, files = extract_length_scans(src, commits, files)
    usb, files = extract_usb_scan(src, commits, files)

    groups = {
        "glm-rows.jsonl": glm + sweeps,
        "dsv41-ctxscan.jsonl": scans,
        "qwen-usb-scan.jsonl": usb,
    }

    if args.check:
        for name, rows in groups.items():
            print("## " + name)
            for r in rows:
                print("  %-52s %-13s %7s %8.1f %s" % (
                    r["id"], r["store"], r["prompt_tokens"], r["prefill_tok_s"],
                    r.get("date") or ""))
        return

    out = os.path.abspath(args.out)
    for name, rows in groups.items():
        write_jsonl(os.path.join(out, name), rows)

    manifest = {
        "sources": [
            {"repo": repo, "commit": commit, "path": path, "sha256": digest}
            for (repo, commit, path, digest) in sorted(files)
        ],
        "rows": {name: len(rows) for name, rows in groups.items()},
    }
    with open(os.path.join(out, "sources.json"), "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    print("wrote %s: %s" % (out, ", ".join("%s=%d" % (n, len(r)) for n, r in groups.items())))


if __name__ == "__main__":
    main()
