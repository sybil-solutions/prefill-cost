# Row schema

One JSON object per line in `data/raw/<family>.jsonl`. One row is one measured **prefill**:
a prompt of `prompt_tokens` tokens pushed through a model, reported as `prefill_tok_s`.
Never average, never combine two runs into one row, never fill a field the source does not state.

| field | type | required | meaning |
|---|---|---|---|
| `id` | str | yes | `<dataset>/<run>@<tokens>`, unique within the file. Append `~a` / `~b` for repeat runs of the same config and length. |
| `family` | str | yes | `qwen`, `deepseek`, `glm`, or `other`. |
| `dataset` | str | yes | Campaign or repo the row belongs to, e.g. `qwen-serve-b70-offload`, `spark-stack`. Rows in the same dataset share one protocol and one host. |
| `source.repo` | str | yes | `sybil-solutions/<name>`. |
| `source.commit` | str | yes | Full 40-char commit sha you extracted from (never a branch name). |
| `source.path` | str | yes | Path inside that repo of the file the number came from. |
| `source.run_id` | str | no | Run/arm label as the source names it. |
| `provenance` | str | yes | `result-json` (machine-read JSON), `registry-run` (curated run record), `sweep-json` (bench sweep output), or `transcribed` (read off a Markdown table). |
| `date` | str | yes | `YYYY-MM-DD` the run happened, from the source. |
| `model` | str | yes | Model name, e.g. `Qwen3.8-27B`. |
| `quant` | str | no | Quantization/format as stated, e.g. `AWQ-INT4 + DSpark`, `EXL3 3.05 bpw`. |
| `gpu` | str | yes | e.g. `1x RTX 3090 24 GB`, `2x DGX Spark GB10`, `1x Intel Arc Pro B70 32 GB`. |
| `cpu` | str | no | Host CPU as stated. |
| `host` | str | no | Host nickname the source uses. |
| `store` | str | yes | Where the weights that prefill must read actually live, from the vocabulary below. |
| `store_detail` | str | yes | One sentence: drives, layout, link. State the source's own numbers when it gives them. |
| `ram_cap_gb` | int \| null | no | Container/host RAM cap the run was constrained to, when the source states it. |
| `offload` | bool | yes | Does prefill read weights from somewhere other than GPU VRAM (NVMe, host RAM, another host)? |
| `mode` | str | no | Engine mode/config name as the source gives it. |
| `protocol` | str | yes | How prefill was measured, quoted from the source (`bench/sweep.py --prefill 8192 32768 ...`, `vllm bench serve`, `TTFT method`, etc.). This is what makes rows comparable or not. |
| `prompt_tokens` | int | yes | Prompt length in tokens. |
| `prefill_tok_s` | float | yes | Prefill speed as measured. |
| `ttft_s` | float \| null | no | Time to first token, if the source states it. |
| `decode_tok_s` | float \| null | no | Decode speed recorded in the same run, as contrast. This dataset is about prefill; keep the decode number only when the source pairs it with that prefill measurement, so a reader can see whether a tier change moved prefill and decode the same way. |
| `concurrency` | int \| null | no | Requests in flight during the measurement (1 = single stream). |
| `io` | object \| null | no | Storage counters for the prefill phase, if the source has them: `read_gbps`, `iops`, `read_latency_ms`, `pct_of_ceiling`, `avg_request_kb`. |
| `notes` | str \| null | no | Anything a reader needs to not over-read the number: aborted arm, different image, tiny prompt, etc. |

## `store` vocabulary

Use exactly one of these. Anything more specific belongs in `store_detail`.

- `vram-only`: everything fits in GPU VRAM; prefill reads no tier.
- `ram-all`: routed experts pinned in host RAM, streamed to VRAM over PCIe.
- `nvme-single`: one drive in a motherboard M.2 slot.
- `nvme-raid0`: several drives in a striped array, usually on a PCIe carrier card.
- `nvme-capped`: an array with its bandwidth capped to emulate a slower tier.
- `unified-memory`: one box with a large shared memory pool (DGX Spark, M3 Ultra, Strix Halo).
- `usb-or-oculink`: the tier lives on a drive or a peer host over a peripheral link.
- `disk-engram`: weights in VRAM/RAM but auxiliary tables memory-mapped on disk (say so in `store_detail`).
- `unknown`: the source does not state it.

## Evidence rules

1. A number with no file behind it does not go in. Cite `source.path`, and `source.commit` must be the
   commit you actually read.
2. Markdown tables are acceptable only with `provenance: "transcribed"` and the table's section heading in
   `notes`. Prefer JSON whenever the repo has it.
3. Multiple prompt lengths from one run = multiple rows. Multiple streams = `concurrency: 4`, and the
   prefill speed is whatever the source reports for that row (state it in `notes` if the source reports a
   single prefill figure for all concurrency levels).
4. If a repo reports prefill only inside a paragraph of prose, transcribe it and quote the sentence in `notes`.
5. If a run was aborted, lossy, or ran a different image than its neighbours, say so in `notes`.
6. Do not convert units, do not recompute another repo's tok/s from TTFT unless the source gives both and
   your recomputation matches; if you do compute `ttft_s` from `prompt_tokens / prefill_tok_s`, leave
   `ttft_s: null` and note it. Derived columns are the renderer's job, not the extractor's.
