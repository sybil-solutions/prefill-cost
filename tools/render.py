#!/usr/bin/env python3
"""Render the README tables and the site from data/prefill.jsonl.

Usage: tools/render.py [--check]

Writes between the <!-- BEGIN:x --> / <!-- END:x --> markers in README.md and
regenerates docs/index.html. Never edits prose outside the markers.
"""

import argparse
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data", "prefill.jsonl")

# The controlled arms of campaign N129 (2026-10-08, one host, one image).
TIER_RUNS = {
    "8k": ["glm53-n129-q-single-55g", "glm53-n129-q-cap7-55g", "glm53-n129-q-raid-55g",
           "glm53-n129-q-single-16g", "glm53-n129-q-cap7-16g", "glm53-n129-q-raid-16g"],
    "32k": ["glm53-n129-f-single-16g", "glm53-n129-g-cap7-16g", "glm53-n129-g-raid-16g",
            "glm53-55g-pfon-n129"],
}
STORE_ORDER = ["nvme-single", "nvme-capped", "nvme-raid0", "ram-all",
               "usb-or-oculink", "unified-memory", "vram-only", "disk-engram", "unknown"]
STORE_NAME = {
    "nvme-single": "1x M.2 drive",
    "nvme-capped": "4x array, capped to 1 drive's bandwidth",
    "nvme-raid0": "4x M.2 array in RAID0",
    "ram-all": "all experts in host RAM",
    "usb-or-oculink": "drive on the far side of USB",
    "unified-memory": "one shared memory pool",
    "vram-only": "everything in VRAM",
    "disk-engram": "auxiliary tables on disk",
    "unknown": "not stated",
}


def load():
    with open(DATA) as f:
        return [json.loads(line) for line in f if line.strip()]


def num(v, nd=0):
    if v is None:
        return "-"
    if isinstance(v, float):
        return ("%%.%df" % nd) % v
    return ("%d" % v) if nd == 0 else ("%%.%df" % nd) % v


def ttft(row):
    """Time to first token: stated by the source, else prompt/prefill."""
    if row.get("ttft_s"):
        return row["ttft_s"], ""
    return row["prompt_tokens"] / row["prefill_tok_s"], "*"


def ms_per_1k(row):
    return 1e6 / row["prefill_tok_s"]


# ---------------------------------------------------------------- selectors


def by_run(rows, run_ids):
    out = []
    for rid in run_ids:
        for r in rows:
            if r["source"].get("run_id") == rid:
                out.append(r)
    return out


def tier_rows(rows, key):
    return by_run(rows, TIER_RUNS[key])


def scan_rows(rows, dataset):
    """One length scan: the single run in that dataset that covers the most lengths.

    Marginal cost is only meaningful inside one run, so rows from different arms are never
    mixed into one scan.
    """
    runs = {}
    for r in rows:
        if r["dataset"] != dataset:
            continue
        runs.setdefault(r["source"].get("run_id") or "unstated", []).append(r)
    if not runs:
        return []
    group = max(runs.values(), key=lambda rs: len({r["prompt_tokens"] for r in rs}))
    got = {}
    for r in group:
        pt = r["prompt_tokens"]
        # A/B repeats of one length: keep the faster (the source reports both runs).
        if pt not in got or r["prefill_tok_s"] > got[pt]["prefill_tok_s"]:
            got[pt] = r
    return [got[k] for k in sorted(got)]


def dataset_rows(rows):
    """One summary row per dataset: the fastest prefill it recorded, and its range."""
    groups = {}
    for r in rows:
        groups.setdefault(r["dataset"], []).append(r)
    out = []
    for name, rs in groups.items():
        single = [r for r in rs if r.get("concurrency") in (None, 1)] or rs
        best = max(single, key=lambda r: r["prefill_tok_s"])
        lengths = sorted({r["prompt_tokens"] for r in rs})
        stores = [s for s in STORE_ORDER if any(r["store"] == s for r in rs)]
        models = sorted({r["model"] for r in rs})
        families = sorted({r["family"] for r in rs})
        out.append({
            "dataset": name,
            "family": families[0] if len(families) == 1 else "mixed",
            "model": models[0] if len(models) == 1
                     else "%s +%d more" % (best["model"], len(models) - 1),
            "host": rs[0].get("host") or "-",
            "stores": ", ".join(STORE_NAME[s] for s in stores),
            "lengths": "%s-%s" % (min(lengths), max(lengths)) if len(lengths) > 1
                       else str(lengths[0]),
            "n": len(rs),
            "best": best["prefill_tok_s"],
            "best_at": best["prompt_tokens"],
            "protocol": rs[0]["protocol"].split(":")[0].split(";")[0][:70],
        })
    out.sort(key=lambda r: (-r["best"], r["dataset"]))
    return out


# ---------------------------------------------------------------- tables


def table_tier(rows):
    head = ["store", "RAM cap", "prefill tok/s", "TTFT s", "ms / 1k prompt tokens",
            "decode C1 tok/s", "NVMe read GB/s", "share of link ceiling",
            "avg read latency ms", "stall ms / token"]
    body = []
    for r in sorted(rows, key=lambda r: (r["ram_cap_gb"] or 0,
                                         STORE_ORDER.index(r["store"]))):
        t, _ = ttft(r)
        io = r.get("io") or {}
        body.append([
            STORE_NAME[r["store"]],
            ("%d GB" % r["ram_cap_gb"]) if r["ram_cap_gb"] else "unstated",
            num(r["prefill_tok_s"], 1),
            num(t, 2),
            num(ms_per_1k(r), 2),
            num(r.get("decode_tok_s"), 2),
            num(io.get("read_gbps"), 2),
            ("%.0f%%" % io["pct_of_ceiling"]) if io.get("pct_of_ceiling") else "-",
            num(io.get("read_latency_ms"), 2),
            num(io.get("stall_ms_per_token"), 1),
        ])
    return head, body


def table_scan(rows):
    head = ["prompt tokens", "prefill tok/s", "TTFT s", "ms / 1k tokens",
            "marginal tok/s", "marginal ms / 1k tokens"]
    body = []
    prev = None
    for r in rows:
        t, _ = ttft(r)
        marg = marg_ms = "-"
        if prev is not None:
            dt = t - prev[1]
            if dt > 0:
                marg = num((r["prompt_tokens"] - prev[0]) / dt, 0)
                marg_ms = num(dt * 1e6 / (r["prompt_tokens"] - prev[0]), 1)
        body.append([
            "{:,}".format(r["prompt_tokens"]),
            num(r["prefill_tok_s"], 1),
            num(t, 1),
            num(ms_per_1k(r), 1),
            marg, marg_ms,
        ])
        prev = (r["prompt_tokens"], t)
    return head, body


def table_datasets(rows):
    head = ["dataset", "family", "model", "host", "storage", "prompt tokens",
            "rows", "fastest prefill tok/s", "at tokens", "ms / 1k tokens"]
    body = [[d["dataset"], d["family"], d["model"], d["host"], d["stores"],
             d["lengths"], str(d["n"]), num(d["best"], 1), str(d["best_at"]),
             num(1e6 / d["best"], 0)]
            for d in dataset_rows(rows)]
    return head, body


def table_provenance(rows):
    """One row per source repo. The per-file manifest is data/sources.json."""
    per = {}
    for r in rows:
        src = r["source"]
        e = per.setdefault(src["repo"], {"commits": set(), "files": set(), "rows": 0})
        e["commits"].add(src["commit"])
        e["files"].add(src["path"])
        e["rows"] += 1
    head = ["repo", "commits", "files", "rows"]
    body = []
    for repo, e in sorted(per.items(), key=lambda kv: (-kv[1]["rows"], kv[0])):
        commits = sorted(e["commits"])
        label = "`%s`" % commits[0][:10] if len(commits) == 1 else \
                "`%s` +%d" % (commits[0][:10], len(commits) - 1)
        body.append(["[%s](https://github.com/%s)" % (repo.split("/")[-1], repo),
                     label, str(len(e["files"])), str(e["rows"])])
    return head, body


# ---------------------------------------------------------------- writers


def highlights_md():
    return "\n".join("- **%s** %s" % (t, b) for t, b in HIGHLIGHTS)


def highlights_html():
    out = []
    for t, b in HIGHLIGHTS:
        body, chunk, inside = [], "", False
        for ch in b:
            if ch == "`":
                body.append("<code>%s</code>" % chunk if inside else chunk)
                chunk, inside = "", not inside
            else:
                chunk += ch
        body.append("<code>%s</code>" % chunk if inside else chunk)
        out.append("<li><strong>%s</strong> %s</li>" % (t, "".join(body)))
    return "<ul>\n" + "\n".join(out) + "\n</ul>"


def md_table(head, body):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    for row in body:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def html_table(head, body):
    out = ["<table>", "<thead><tr>" +
           "".join("<th>%s</th>" % h for h in head) + "</tr></thead>", "<tbody>"]
    for row in body:
        cells = []
        for i, c in enumerate(row):
            cls = ' class="n"' if i > 0 and c not in ("-", "") else ""
            cells.append("<td%s>%s</td>" % (cls, c))
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def replace_block(text, name, payload):
    begin, end = "<!-- BEGIN:%s -->" % name, "<!-- END:%s -->" % name
    if begin not in text or end not in text:
        raise SystemExit("README.md is missing the %s markers" % name)
    pre, rest = text.split(begin, 1)
    _, post = rest.split(end, 1)
    return pre + begin + "\n" + payload + "\n" + end + post



# What the wider set adds. Each bullet is one dataset; the number is that
# dataset's fastest single-stream prefill (see table_datasets).
HIGHLIGHTS = [
    ("7,817 tok/s", "at 32k tokens on DeepSeek-V4.1-Flash with the Engram tables on local "
     "NVMe (`deepseek-v4.1-flash-4x-rtx-pro-6000`, 4x RTX PRO 6000). That repo's metric "
     "includes queueing, so treat it as an upper bound; it still reports 5,845 tok/s at 500k."),
    ("4,397 tok/s", "at 32k, and flat from 8k to 64k (4,143-4,397), for Qwen3.8-Flash-Next "
     "MXFP4-FP8-GPTQ on 2x R9700 with the experts in host RAM "
     "(`local-ai-recipe-kit-submissions-2x-r9700-32gb`, a community submission to the kit)."),
    ("2,664 tok/s", "at 106,294 tokens for Qwen3.8-Flash-Next Q2_0 on one RTX 3090 reading a "
     "Strata NVMe expert pack (`local-ai-registry-gates-rtx-3090-24gb`). The IQ3_S pack on the "
     "same prompt measures 2,174."),
    ("2,421 tok/s", "at ~4k for Qwen3.8-27B EXL3 on one Arc Pro B70 with W8A8 int8 prefill "
     "linears, against 1,666 for the same card without them (`trellis-serve-xpu-b70-int8` vs "
     "`trellis-serve-xpu-b70`). On a VRAM-resident model, prefill is a kernel problem, not a "
     "storage one."),
    ("1,015 tok/s", "at 981,841 tokens, 967 s to first token, for GLM-5.3-Flash on 2x DGX Spark "
     "(`glm53-tensorfold-2x-spark`). The only sub-million-token prefill in the set, and the "
     "clearest picture of what a million-token prompt costs."),
    ("70 tok/s", "at 106k tokens on a Radeon 8060S iGPU (`local-ai-registry-gates-radeon-8060s-64gb`), "
     "against 2,664 for the 3090 above. Same prompt shape, 38x apart."),
]

SITE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Prefill cost when the weights do not fit</title>
<style>
:root { color-scheme: light dark; }
body { max-width: 74ch; margin: 2.5rem auto; padding: 0 1.1rem;
       font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       color: #1a1a1a; background: #fdfdfc; }
@media (prefers-color-scheme: dark) {
  body { color: #d8d8d4; background: #141414; }
  a { color: #a5c8ff; }
  th, td { border-color: #333 !important; }
  thead th { background: #1d1d1d !important; }
  code { background: #202020 !important; }
}
h1 { font-size: 1.5rem; line-height: 1.25; margin: 0 0 .3rem; font-weight: 650; }
h2 { font-size: 1.05rem; margin: 2.2rem 0 .5rem; padding-bottom: .25rem;
     border-bottom: 1px solid #e3e3e0; font-weight: 620; }
h3 { font-size: .95rem; margin: 1.4rem 0 .3rem; font-weight: 600; }
p, li { margin: .5rem 0; }
.sub { color: #666; font-size: .9rem; margin-bottom: 1.6rem; }
a { color: #1a4fd6; text-decoration: none; }
a:hover { text-decoration: underline; }
code { font: 12.5px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
       background: #f2f2ef; padding: .08em .3em; border-radius: 2px; }
pre { background: #f2f2ef; padding: .7rem .8rem; overflow-x: auto;
      font: 12.5px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
table { border-collapse: collapse; width: 100%; margin: .8rem 0 1.1rem;
        font-size: 13px; }
th, td { border: 1px solid #e3e3e0; padding: .3rem .45rem; text-align: left;
         vertical-align: top; }
thead th { background: #f7f7f4; font-weight: 600; }
td.n { text-align: right; font-variant-numeric: tabular-nums;
       font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
.note { font-size: .85rem; color: #666; }
nav { font-size: .9rem; margin-bottom: 1.8rem; }
nav a { margin-right: .9rem; }
</style></head><body>
<h1>What prefill costs when the weights do not fit</h1>
<p class="sub">Every measured prefill in the sybil-solutions campaigns, with the storage tier that
served it. __NROWS__ rows from __NDATASETS__ datasets, __NMODELS__ models.
Generated from <code>data/prefill.jsonl</code>; every row cites the file it came from.</p>

<nav><a href="#tier">storage tier</a><a href="#length">prompt length</a>
<a href="#datasets">datasets</a><a href="#caveats">caveats</a>
<a href="#reproduce">reproduce</a></nav>

<h2 id="tier">1. The same prompt, four storage tiers</h2>
<p>GLM-5.3-Flash EXL3 3.05 bpw on one RTX 3090, campaign N129, 2026-10-08. Identical image,
identical settings, only the store changes. The 7 GB/s arms are the same four-drive array with
its bandwidth capped to one drive's, so they separate <em>bandwidth</em> from <em>drive count</em>.</p>
__TABLE_TIER_8K__
<p class="note">__NOTE_TIER_8K__</p>

<h2 id="length">2. What the next thousand prompt tokens cost</h2>
<p>Prefill speed is not constant in prompt length: the first chunk pays for warm-up, the
marginal chunks run near the GPU's own rate. <em>Marginal tok/s</em> is the speed of the tokens
added over the row above, computed from the TTFT column.</p>
__SECTIONS_LENGTH__

<h2 id="rest">3. What the wider set adds</h2>
<p>Thirty-four datasets, most of them one host and one protocol each. These bullets are the
fastest single-stream prefill in each of the ones worth singling out.</p>
__HIGHLIGHTS__
<h3 id="datasets">All datasets</h3>
__TABLE_DATASETS__

<h2 id="caveats">4. Caveats</h2>
<ul>
<li>Prefill numbers are only comparable inside one <code>dataset</code>: protocols differ
(sweep medians vs one request per length), and so do hosts.</li>
<li>Arms of one campaign often ran hours apart. On this host that drifts up to ~17 %, so a
few-percent gap between arms in different sessions is not a result.</li>
<li><code>transcribed</code> rows were read off a Markdown table by hand; everything else is
machine-read from the source repo's JSON.</li>
<li>Time to first token marked with an asterisk is computed as prompt tokens / prefill tok/s,
because the source reports only the rate.</li>
</ul>

<h2 id="reproduce">5. Reproduce</h2>
<pre>git clone https://github.com/sybil-solutions/prefill-cost &amp;&amp; cd prefill-cost
git clone --depth 1 https://github.com/sybil-solutions/moetier ../moetier &amp;&amp; \
git clone --depth 1 https://github.com/sybil-solutions/glm-flash-lite ../glm-flash-lite &amp;&amp; \
git clone --depth 1 https://github.com/sybil-solutions/dsv41-flash-offload ../dsv41-flash-offload &amp;&amp; \
git clone --depth 1 https://github.com/sybil-solutions/qwen38-flash-next-3090-usb-offload ../qwen38-flash-next-3090-usb-offload
python3 tools/extract.py --sources ..   # staging rows
python3 tools/merge.py                  # validate + data/prefill.jsonl
python3 tools/render.py                 # README tables + docs/index.html</pre>

<h2 id="provenance">Provenance</h2>
__TABLE_PROVENANCE__
</body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="render but do not write")
    args = ap.parse_args()

    rows = load()

    t8_head, t8_body = table_tier(tier_rows(rows, "8k"))
    scans = [("dsv41-flash-offload", "EXL3 3.0 bpw on one 3090, experts pinned in host RAM, run D119"),
             ("qwen38-flash-next-3090-usb-offload", "EXL3 3.05 bpw on one 3090 whose experts live on a Mac mini's SSD over USB4, config C")]

    datasets_table = html_table(*table_datasets(rows))
    prov_table = html_table(*table_provenance(rows))

    sections_md, sections_html = [], []
    for dataset, blurb in scans:
        rs = scan_rows(rows, dataset)
        if not rs:
            continue
        head, body = table_scan(rs)
        sections_md.append("**%s** (%s)\n\n%s" % (rs[0]["model"], blurb, md_table(head, body)))
        sections_html.append("<h3>%s</h3>\n<p>%s</p>\n%s" % (
            rs[0]["model"], blurb, html_table(head, body)))

    page = (SITE
            .replace("__NROWS__", "{:,}".format(len(rows)))
            .replace("__NDATASETS__", str(len({r["dataset"] for r in rows})))
            .replace("__NMODELS__", str(len({r["model"] for r in rows})))
            .replace("__TABLE_TIER_8K__", html_table(t8_head, t8_body))
            .replace("__NOTE_TIER_8K__",
                     "8k prompt, C1. TTFT = 8192 / prefill tok/s. I/O columns are the "
                     "prefill phase's device counters.")
            .replace("__SECTIONS_LENGTH__", "\n".join(sections_html))
            .replace("__HIGHLIGHTS__", highlights_html())
            .replace("__TABLE_DATASETS__", datasets_table)
            .replace("__TABLE_PROVENANCE__", prov_table))

    readme_path = os.path.join(ROOT, "README.md")
    readme = open(readme_path).read()
    readme = replace_block(readme, "tier-8k", md_table(t8_head, t8_body))
    readme = replace_block(readme, "highlights", highlights_md())
    readme = replace_block(readme, "datasets", md_table(*table_datasets(rows)))
    readme = replace_block(readme, "provenance", md_table(*table_provenance(rows)))
    readme = replace_block(readme, "length", "\n\n".join(sections_md))

    if args.check:
        print(page[:1200])
        return
    with open(readme_path, "w") as f:
        f.write(readme)
    site_dir = os.path.join(ROOT, "docs")
    os.makedirs(site_dir, exist_ok=True)
    with open(os.path.join(site_dir, "index.html"), "w") as f:
        f.write(page)
    print("rendered README.md and docs/index.html from %d rows" % len(rows))


if __name__ == "__main__":
    main()
