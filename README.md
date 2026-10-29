# prefill-cost

What a prompt costs to prefill when the weights do not fit in the GPU.

Prompt processing is the expensive half of long-context inference on a small card, and it is the
half that reads from wherever the experts actually live — VRAM, host RAM, one NVMe drive, or a
striped array. This repo collects every prefill measurement the sybil-solutions campaigns have
made, normalized to one row per (run, prompt length), each row citing the file it came from. The
point is not a leaderboard: it is to show what each storage tier actually buys, and where prefill
stops being storage-bound at all.

`data/prefill.jsonl` is the dataset. `data/summary.json` counts it. `docs/schema.md` is the row
contract. Numbers below are generated from the dataset by `tools/render.py`.

## Short version

- **Four M.2 drives in RAID0 prefill 2.7x faster than one M.2 drive** at a 16 GB RAM cap
  (650 vs 242 tok/s, 8k prompt, GLM-5.3-Flash on one RTX 3090, campaign N129, same day and
  image), and 1.65x at a 55 GB cap (664 vs 402).
- **Prefill buys bandwidth; decode buys drive count.** Capping that array to a single drive's
  7 GB/s puts prefill back at the single drive's rate (393 vs 402 tok/s at 55 GB, 254 vs 242 at
  16 GB) — so prefill's gain is bandwidth, and nothing else. The same capped array still decodes
  29-41 % faster than the real single drive (14.34 vs 10.16 tok/s at 55 GB, 4.52 vs 3.50 at
  16 GB). Prefill keeps tens of thousands of reads in flight and saturates the link; decode has a
  few dozen per token and pays per-read latency, where 0.8-1.3 ms beats 4.8-7.0 ms. If you are
  sizing an array for prefill alone, buy bandwidth; the queue-depth win is a decode win.
- **Prefill gets cheaper per token as the prompt grows.** On a single 990 PRO with experts in
  host RAM, DeepSeek-V4.1-Flash goes 589 tok/s at 8k to 813 at 65k, and the marginal cost of the
  tokens between 8k and 32k is ~866 tok/s. The first chunks pay warm-up; the rest run near the
  card's own rate. Absolute TTFT still climbs to **342 s for a 261k-token prompt**.
- **Prefill costs about 8x more on the small box.** Per 1,000 prompt tokens: 128 ms on 4x
  RTX PRO 6000, 1,027 ms on one 3090 with the RAID0 store at 32k (and 1,506 ms at 8k with the
  same store at a 55 GB cap). A 100k-token prompt is ~13 s of prefill on the big box against
  ~103 s on the little one — before it generates a token.
- **When prefill is already GPU-bound, storage stops mattering.** On an Arc Pro B70 whose experts
  sit in RAM, the same 7 GB/s cap costs nothing measurable (975 vs 1064 tok/s, inside run-to-run
  spread) because prefill reads only 4-5 GB/s, 16-20 % of the link. A faster array would buy that
  box nothing.

## 1. The same prompt, four storage tiers

Campaign N129 (2026-10-08) is the controlled comparison: one host, one image, one set of engine
settings, and only the store changed. `4x M.2 array, capped to 1 drive's bandwidth` is the same
array with `docker --device-read-bps` at 7.0 GB/s — the arm that separates bandwidth from drive
count.

<!-- BEGIN:tier-8k -->
| store | RAM cap | prefill tok/s | TTFT s | ms / 1k prompt tokens | decode C1 tok/s | NVMe read GB/s | share of link ceiling | avg read latency ms | stall ms / token |
|---|---|---|---|---|---|---|---|---|---|
| 1x M.2 drive | 16 GB | 242.2 | 33.82 | 4128.82 | 3.50 | 6.46 | 87% | 7.04 | - |
| 4x array, capped to 1 drive's bandwidth | 16 GB | 254.3 | 32.21 | 3932.36 | 4.52 | 6.83 | 98% | 1.88 | - |
| 4x M.2 array in RAID0 | 16 GB | 649.7 | 12.61 | 1539.17 | 12.79 | 17.26 | 66% | 0.72 | 38.4 |
| 1x M.2 drive | 55 GB | 402.4 | 20.36 | 2485.09 | 10.16 | 6.77 | 91% | 4.84 | 46.6 |
| 4x array, capped to 1 drive's bandwidth | 55 GB | 393.3 | 20.83 | 2542.59 | 14.34 | 6.72 | 96% | 1.31 | 23.5 |
| 4x M.2 array in RAID0 | 55 GB | 663.8 | 12.34 | 1506.48 | 17.43 | 11.18 | 43% | 0.84 | 10.6 |
<!-- END:tier-8k -->

Read the I/O columns next to the speeds, and prefill looks purely bandwidth-limited. The single
drive moves 6.5 GB/s of its 7.45 GB/s ceiling (87 %) at 49k IOPS; the array at 16 GB moves
17.3 GB/s of 26 GB/s (66 %) at 148k IOPS. The capped array sits within 5 % of the real single
drive on prefill at both RAM caps — 254 vs 242 tok/s at 16 GB, 393 vs 402 at 55 GB — which is
what makes it a fair stand-in, and what proves the 4-drive prefill win is the extra bandwidth.

The decode column is where drive count earns more than its bandwidth. At the same 7 GB/s the
array decodes 41 % faster than the single drive at 55 GB (14.34 vs 10.16 tok/s) and 29 % faster
at 16 GB (4.52 vs 3.50), because decode holds only a few dozen reads in flight per token and so
pays the per-read latency: 0.8-1.3 ms on the array against 4.8-7.0 ms on the single drive.

32k prompts: single 990 PRO 338 tok/s, capped array 423, RAID0 957. The gap widens with prompt
length because a longer prefill issues more reads.

## 2. What the next thousand prompt tokens cost

Prefill speed is not a constant, so the useful number is marginal cost. Each block below is one
single-run scan at C1; *marginal tok/s* is the speed of the tokens added over the row above,
derived from TTFT.

<!-- BEGIN:length -->
**DeepSeek-V4.1-Flash** (EXL3 3.0 bpw on one 3090, experts pinned in host RAM, run D119)

| prompt tokens | prefill tok/s | TTFT s | ms / 1k tokens | marginal tok/s | marginal ms / 1k tokens |
|---|---|---|---|---|---|
| 8,192 | 589.1 | 13.9 | 1697.5 | - | - |
| 32,768 | 774.6 | 42.3 | 1291.0 | 866 | 1155.2 |
| 65,536 | 813.2 | 80.6 | 1229.7 | 856 | 1168.5 |
| 131,072 | 791.4 | 165.6 | 1263.6 | 771 | 1297.6 |
| 196,608 | 781.1 | 251.7 | 1280.2 | 761 | 1313.3 |
| 261,000 | 764.1 | 341.6 | 1308.7 | 716 | 1395.8 |

**Qwen3.8-Flash-Next** (EXL3 3.05 bpw on one 3090 whose experts live on a Mac mini's SSD over USB4, config C)

| prompt tokens | prefill tok/s | TTFT s | ms / 1k tokens | marginal tok/s | marginal ms / 1k tokens |
|---|---|---|---|---|---|
| 2,565 | 176.7 | 14.5 | 5659.3 | - | - |
| 8,018 | 417.8 | 19.2 | 2393.5 | 1166 | 857.3 |
| 18,330 | 470.1 | 39.0 | 2127.2 | 521 | 1920.2 |
| 33,842 | 584.2 | 57.9 | 1711.7 | 819 | 1220.8 |
| 97,811 | 560.9 | 175.0 | 1782.8 | 546 | 1830.1 |
| 189,481 | 548.1 | 346.0 | 1824.5 | 536 | 1865.4 |
<!-- END:length -->

## 3. What the wider set adds

The controlled experiment above is GLM-only. The rest of the dataset is breadth: 34 datasets
across DeepSeek, Qwen, GLM, Nemotron, Welp and LFM checkpoints on everything from one 3090 to 4x
RTX PRO 6000, DGX Sparks, Strix Halo, an M-series Mac and a USB4 eGPU. Each bullet below is one
dataset's fastest single-stream prefill.

<!-- BEGIN:highlights -->
- **7,817 tok/s** at 32k tokens on DeepSeek-V4.1-Flash with the Engram tables on local NVMe (`deepseek-v4.1-flash-4x-rtx-pro-6000`, 4x RTX PRO 6000). That repo's metric includes queueing, so treat it as an upper bound; it still reports 5,845 tok/s at 500k.
- **4,397 tok/s** at 32k, and flat from 8k to 64k (4,143-4,397), for Qwen3.8-Flash-Next MXFP4-FP8-GPTQ on 2x R9700 with the experts in host RAM (`local-ai-recipe-kit-submissions-2x-r9700-32gb`, a community submission to the kit).
- **2,664 tok/s** at 106,294 tokens for Qwen3.8-Flash-Next Q2_0 on one RTX 3090 reading a Strata NVMe expert pack (`local-ai-registry-gates-rtx-3090-24gb`). The IQ3_S pack on the same prompt measures 2,174.
- **2,421 tok/s** at ~4k for Qwen3.8-27B EXL3 on one Arc Pro B70 with W8A8 int8 prefill linears, against 1,666 for the same card without them (`trellis-serve-xpu-b70-int8` vs `trellis-serve-xpu-b70`). On a VRAM-resident model, prefill is a kernel problem, not a storage one.
- **1,015 tok/s** at 981,841 tokens, 967 s to first token, for GLM-5.3-Flash on 2x DGX Spark (`glm53-tensorfold-2x-spark`). The only sub-million-token prefill in the set, and the clearest picture of what a million-token prompt costs.
- **70 tok/s** at 106k tokens on a Radeon 8060S iGPU (`local-ai-registry-gates-radeon-8060s-64gb`), against 2,664 for the 3090 above. Same prompt shape, 38x apart.
<!-- END:highlights -->

### Every dataset


<!-- BEGIN:datasets -->
| dataset | family | model | host | storage | prompt tokens | rows | fastest prefill tok/s | at tokens | ms / 1k tokens |
|---|---|---|---|---|---|---|---|---|---|
| deepseek-v4.1-flash-4x-rtx-pro-6000 | deepseek | DeepSeek-V4.1-Flash | - | auxiliary tables on disk, not stated | 512-500000 | 89 | 7816.6 | 32768 | 128 |
| local-ai-recipe-kit-submissions-2x-r9700-32gb | qwen | Qwen3.8-Flash-Next | - | all experts in host RAM | 8192-65536 | 16 | 4396.7 | 32768 | 227 |
| deepseek-v4.1-flash-h200 | deepseek | DeepSeek-V4.1-Flash | - | all experts in host RAM, auxiliary tables on disk | 8192-32980 | 32 | 3197.9 | 32737 | 313 |
| local-ai-recipe-kit-3090 | qwen | Qwen3.8-Flash-Next | omarchy | 1x M.2 drive | 8192-65536 | 4 | 2848.0 | 32768 | 351 |
| local-ai-recipe-kit-targets-rtx-3090-24gb | qwen | Qwen3.8-Flash-Next | - | 1x M.2 drive | 8192-65536 | 4 | 2846.0 | 32768 | 351 |
| local-ai-recipe-kit-submissions-rtx-3090-24gb | qwen | Qwen3.8-Flash-Next | omarchy | 1x M.2 drive | 8192-65536 | 4 | 2836.9 | 32768 | 352 |
| local-ai-registry-gates-rtx-3090-24gb | mixed | Qwen3.8-Flash-Next +3 more | rtx-3090-24gb | 1x M.2 drive, all experts in host RAM, everything in VRAM | 93437-173372 | 11 | 2664.0 | 106294 | 375 |
| trellis-serve-xpu-b70-int8 | qwen | Qwen3.8-27B | - | everything in VRAM | 4096-253952 | 4 | 2421.0 | 4096 | 413 |
| local-ai-registry-gates-dgx-spark-gb10-128gb | mixed | GLM-5.3-Flash +1 more | dgx-spark-gb10-128gb | one shared memory pool | 175457-751923 | 3 | 2094.0 | 751923 | 478 |
| glm53-tensorfold-3x-spark | glm | GLM-5.3-Flash | three DGX Sparks (TP=3, NCCL, KV_POOL_GIB=27) | one shared memory pool | 8212-262169 | 6 | 2064.2 | 32790 | 484 |
| local-ai-registry-gates-rtx-4080-super-16gb | qwen | Welp-35B-A3B | rtx-4080-super-16gb | everything in VRAM | 213570 | 1 | 2028.0 | 213570 | 493 |
| glm53-tensorfold-2x-spark | glm | GLM-5.3-Flash | two DGX Sparks (TP=2 over ConnectX-7 RoCE) | one shared memory pool | 8219-981841 | 7 | 1978.9 | 32790 | 505 |
| trellis-serve-xpu-b70 | qwen | Qwen3.8-27B | - | everything in VRAM | 4185-254125 | 22 | 1666.0 | 4200 | 600 |
| qwen-serve-27b-3090-sglang | qwen | Qwen3.8-27B | - | everything in VRAM | 8741-138452 | 40 | 1615.8 | 17397 | 619 |
| moetier-qwen38fn | qwen | Qwen3.8-Flash-Next | omarchy | 4x array, capped to 1 drive's bandwidth, 4x M.2 array in RAID0 | 8192-32768 | 55 | 1193.4 | 32768 | 838 |
| qwen-serve-27b-b70-llamacpp | qwen | Qwen3.8-27B | - | everything in VRAM | 2500-245000 | 12 | 1164.0 | 2500 | 859 |
| local-ai-registry-gates-intel-arc-pro-b70-32gb | mixed | Qwen3.8-Flash-Next +1 more | intel-arc-pro-b70-32gb | 1x M.2 drive, all experts in host RAM, everything in VRAM | 213575-213612 | 7 | 1087.0 | 213612 | 920 |
| qwen-serve-b70-offload | qwen | Qwen3.8-Flash-Next | omarchy | 4x M.2 array in RAID0 | 942-7406 | 4 | 1029.5 | 7406 | 971 |
| local-ai-recipe-kit-targets-intel-arc-pro-b70-32gb | qwen | Qwen3.8-Flash-Next | - | 1x M.2 drive | 8192-65536 | 4 | 1027.0 | 8192 | 974 |
| local-ai-recipe-kit-b70 | qwen | Qwen3.8-Flash-Next | - | 1x M.2 drive | 8192-65536 | 4 | 1022.7 | 8192 | 978 |
| trellis-serve-xpu-b70-llamacpp | qwen | Qwen3.8-27B | - | everything in VRAM | 7335-58172 | 2 | 999.1 | 7335 | 1001 |
| glm-flash-lite | glm | GLM-5.3-Flash | omarchy | 1x M.2 drive, 4x array, capped to 1 drive's bandwidth, 4x M.2 array in RAID0, all experts in host RAM, not stated | 8192-32768 | 96 | 973.7 | 32768 | 1027 |
| dsv41-flash-offload | deepseek | DeepSeek-V4.1-Flash | - | all experts in host RAM | 8192-261000 | 23 | 813.2 | 65536 | 1230 |
| qwen38-flash-next-3090-usb-offload | qwen | Qwen3.8-Flash-Next | mini | drive on the far side of USB | 2565-189481 | 10 | 584.2 | 33842 | 1712 |
| local-ai-registry-gates-rx-6700-xt-12gb | qwen | Qwen3.5-9B | rx-6700-xt-12gb | everything in VRAM | 52583 | 1 | 559.0 | 52583 | 1789 |
| local-ai-registry-gates-rtx-3060-12gb | qwen | Qwen3.6-35B-A3B +1 more | rtx-3060-12gb | all experts in host RAM, everything in VRAM | 106252-106294 | 2 | 512.0 | 106252 | 1953 |
| framework-research-phase0-strix | qwen | Qwen3.5-122B-A10B-REAP-20 +1 more | Strix Halo Framework Desktop | one shared memory pool | 64-131072 | 33 | 443.0 | 512 | 2257 |
| trellis-serve-xpu-b70-llamacpp-tp2 | qwen | Qwen3.8-27B | - | everything in VRAM | 7379-58352 | 2 | 341.4 | 7379 | 2929 |
| framework-research-rpc-hetero | qwen | Qwen3.5-122B-A10B-REAP-20 | halo (Strix Halo) + <cuda-peer> RTX 3090 over Wi-Fi RPC | drive on the far side of USB, one shared memory pool | 20-8832 | 12 | 320.9 | 8832 | 3117 |
| local-ai-registry-gates-apple-m5-pro-48gb | qwen | Qwen3.8-Flash-Next +1 more | apple-m5-pro-48gb | one shared memory pool | 25740-166625 | 3 | 218.0 | 25740 | 4587 |
| local-ai-registry-gates-apple-m5-pro-24gb | qwen | Qwen3.8-27B | apple-m5-pro-24gb | one shared memory pool | 166655 | 1 | 176.0 | 166655 | 5682 |
| local-ai-registry-gates-x86-64-avx2-cpu | other | LFM2.5-2.6B | x86-64-avx2-cpu | all experts in host RAM | 5552 | 1 | 84.0 | 5552 | 11905 |
| local-ai-registry-gates-radeon-8060s-64gb | qwen | Qwen3.8-27B | radeon-8060s-64gb | one shared memory pool | 106252 | 1 | 70.0 | 106252 | 14286 |
| local-ai-registry-gates-apple-m2-max-96gb | qwen | Qwen3.8-27B | apple-m2-max-96gb | one shared memory pool | 166667 | 1 | 52.0 | 166667 | 19231 |
<!-- END:datasets -->

## 4. Method

- One row = one measured prefill: `prompt_tokens`, `prefill_tok_s`, and where the weights were
  read from. `offload` says whether prefill read anything outside VRAM.
- `provenance` distinguishes machine-read JSON (`result-json`, `registry-run`, `sweep-json`) from
  numbers read off a Markdown table by hand (`transcribed`). Hand-transcribed rows are the
  minority and are marked in the dataset.
- `protocol` quotes how prefill was measured, because it is what makes rows comparable or not:
  a median of 3 sweep reps is not the same measurement as one request per length.
- TTFT in the generated tables is the source's own where it states one; otherwise it is
  `prompt_tokens / prefill_tok_s`, marked with an asterisk.

## 5. Caveats

- **Compare inside a dataset, not across.** Protocols, host CPUs and engine versions differ.
- **Session drift.** Arms that ran hours apart on this host differ by up to ~17 %. A few-percent
  gap between different sessions is not a result; the N129 table above is one session per prompt
  size, and the 8k and 32k blocks ran in different sessions.
- **Capped arms are emulations.** `nvme-capped` reproduces one drive's bandwidth on a 4-drive
  array. It matches the real single drive within 5 % on prefill, but it is not the same hardware,
  and its read latency (1.3 ms) is nothing like the single drive's (4.8-7.0 ms).
- **Aborted and lossy arms are still in the data.** The `notes` field says so; nothing was
  silently dropped.
- Storage ceilings are the ones the source states for its own array (26.0 GB/s RAID0,
  7.45 GB/s single drive). They are not theoretical interface limits.
- **The `local-ai-registry-gates-*` datasets are not sweeps.** They are six-gate validation runs;
  their prefill figure comes from one context-gate prompt with no TTFT instrument, so it is
  `prompt_tokens / whole-request seconds`, a lower bound on prefill speed. They are in here
  because they are the only measurements on a 3060, 4080 Super, 6700 XT, M2 Max and M5 Pro.
- **Some dates are commit dates.** Rows carrying `date_is_commit: true` come from repos that state
  no measurement date; the date is the commit that landed the table, which makes it an upper
  bound on when the run happened.
- **Some prompt lengths are nominal.** Transcribed tables label a cell "8K" while the harness sent
  8,187 tokens. Where a dataset mixes nominal and measured lengths, `prompt_tokens` holds what the
  source states and `notes` says which kind it is.
- **The datasets table's "fastest" column is single-stream only** (`concurrency` 1 or unstated).
  Aggregate-throughput cells exist in the data (DeepSeek's C2/C4/C6/C8 cells, for one) and are
  higher; they are not per-stream prefill.

## 6. Reproduce

```bash
git clone https://github.com/sybil-solutions/prefill-cost && cd prefill-cost
for r in moetier glm-flash-lite dsv41-flash-offload qwen38-flash-next-3090-usb-offload; do
  git clone --depth 1 https://github.com/sybil-solutions/$r ../$r
done
python3 tools/extract.py --sources ..   # staging rows, machine-read
python3 tools/merge.py                  # schema check + data/prefill.jsonl
python3 tools/render.py                 # README tables + site/index.html
```

`tools/extract.py` covers the repos whose numbers are machine-readable in a stable place. The
remaining rows were harvested per family into `data/raw/{qwen,deepseek,other}.jsonl` under the
same schema; `tools/merge.py` validates and folds all of them together.

## Provenance

Every row names its repo, commit and file; `data/staging/sources.json` carries the sha256 of each
file that was machine-read.

<!-- BEGIN:provenance -->
| repo | commit | file | rows |
|---|---|---|---|
| [deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000) | `ce69105cb581` | `results/capacity-2800000.json` | 1 |
| [deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000) | `ce69105cb581` | `results/local-nvme-dspark-summary.json` | 45 |
| [deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000) | `ce69105cb581` | `results/remote-historical-summary.json` | 3 |
| [deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000) | `ce69105cb581` | `results/summary.json` | 40 |
| [deepseek-v4.1-flash-h200](https://github.com/sybil-solutions/deepseek-v4.1-flash-h200) | `8a0104bc9f86` | `evidence/fixed-vision/summary.json` | 27 |
| [deepseek-v4.1-flash-h200](https://github.com/sybil-solutions/deepseek-v4.1-flash-h200) | `8a0104bc9f86` | `evidence/ram-benchmark.json` | 1 |
| [deepseek-v4.1-flash-h200](https://github.com/sybil-solutions/deepseek-v4.1-flash-h200) | `8a0104bc9f86` | `evidence/ssd-benchmark.json` | 1 |
| [deepseek-v4.1-flash-h200](https://github.com/sybil-solutions/deepseek-v4.1-flash-h200) | `8a0104bc9f86` | `evidence/tables/measured.json` | 2 |
| [deepseek-v4.1-flash-h200](https://github.com/sybil-solutions/deepseek-v4.1-flash-h200) | `8a0104bc9f86` | `evidence/tiered/cold-benchmark.json` | 1 |
| [dsv41-flash-offload](https://github.com/sybil-solutions/dsv41-flash-offload) | `08802b6fe001` | `results/D030/sweep.json` | 2 |
| [dsv41-flash-offload](https://github.com/sybil-solutions/dsv41-flash-offload) | `08802b6fe001` | `results/D061/sweep.json` | 2 |
| [dsv41-flash-offload](https://github.com/sybil-solutions/dsv41-flash-offload) | `08802b6fe001` | `results/D107/sweep.json` | 2 |
| [dsv41-flash-offload](https://github.com/sybil-solutions/dsv41-flash-offload) | `08802b6fe001` | `results/D117/sweep.json` | 2 |
| [dsv41-flash-offload](https://github.com/sybil-solutions/dsv41-flash-offload) | `08802b6fe001` | `results/D119/ctx_scan.json` | 5 |
| [dsv41-flash-offload](https://github.com/sybil-solutions/dsv41-flash-offload) | `08802b6fe001` | `results/D119/ctx_scan_261k.json` | 1 |
| [dsv41-flash-offload](https://github.com/sybil-solutions/dsv41-flash-offload) | `08802b6fe001` | `results/D130.md` | 9 |
| [framework-research](https://github.com/sybil-solutions/framework-research) | `7eb74068e3ca` | `benchmarks/missions/34-rpc-hetero-inference/results_raw.jsonl` | 12 |
| [framework-research](https://github.com/sybil-solutions/framework-research) | `7eb74068e3ca` | `documentation/phase-0-strix-benchmarks.md` | 33 |
| [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) | `6fe077da79a4` | `results/C056/sweep.json` | 2 |
| [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) | `6fe077da79a4` | `results/G066a/sweep.json` | 2 |
| [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) | `6fe077da79a4` | `results/G067/sweep.json` | 2 |
| [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) | `6fe077da79a4` | `results/N119-S2a-55g/sweep.json` | 2 |
| [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) | `6fe077da79a4` | `results/N119-S3a-55g/sweep.json` | 2 |
| [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) | `6fe077da79a4` | `results/N119-S3b2-55g/sweep.json` | 2 |
| [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) | `6fe077da79a4` | `results/N129-pfon-55g/sweep.json` | 2 |
| [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) | `6fe077da79a4` | `results/N129-s16-r16-16g/sweep.json` | 1 |
| [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) | `6fe077da79a4` | `results/N129-s16-r48-16g/sweep.json` | 1 |
| [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) | `6fe077da79a4` | `results/P001-image-smoke/sweep.json` | 2 |
| [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) | `6fe077da79a4` | `results/P002b-stream-fix-smoke/sweep.json` | 2 |
| [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) | `6fe077da79a4` | `results/P003-nvme-image-smoke/sweep.json` | 2 |
| [glm53-flash-tensorfold-4x-spark](https://github.com/sybil-solutions/glm53-flash-tensorfold-4x-spark) | `16a92c85d6c4` | `README.md` | 13 |
| [local-ai-recipe-kit](https://github.com/sybil-solutions/local-ai-recipe-kit) | `ef9782ce0c56` | `reference/best_known.json` | 8 |
| [local-ai-recipe-kit](https://github.com/sybil-solutions/local-ai-recipe-kit) | `ef9782ce0c56` | `submissions/2x-r9700-32gb/2026-09-29-benstadin-bprime-fp8.json` | 4 |
| [local-ai-recipe-kit](https://github.com/sybil-solutions/local-ai-recipe-kit) | `ef9782ce0c56` | `submissions/2x-r9700-32gb/2026-09-29-benstadin-bprime.json` | 4 |
| [local-ai-recipe-kit](https://github.com/sybil-solutions/local-ai-recipe-kit) | `ef9782ce0c56` | `submissions/2x-r9700-32gb/2026-09-29-benstadin-m2-fp8.json` | 4 |
| [local-ai-recipe-kit](https://github.com/sybil-solutions/local-ai-recipe-kit) | `ef9782ce0c56` | `submissions/2x-r9700-32gb/2026-09-29-benstadin-m2.json` | 4 |
| [local-ai-recipe-kit](https://github.com/sybil-solutions/local-ai-recipe-kit) | `ef9782ce0c56` | `submissions/rtx-3090-24gb/2026-09-29-0xSero.json` | 4 |
| [local-ai-recipe-kit](https://github.com/sybil-solutions/local-ai-recipe-kit) | `ef9782ce0c56` | `targets/qwen3.8-flash-next-offload.md` | 8 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/apple-m2-max-96gb.qwen3.8-27b.mlx-vlm-qwen3.8-27b-4bit-mtp-200k-m2-max.200k.20261006T084645.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/apple-m5-pro-24gb.qwen3.8-27b.mlx-vlm-qwen3.8-27b-3bit-mtp-200k.200k.20261006T032555.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/apple-m5-pro-48gb.qwen3.8-27b.mtplx-qwen3.8-27b-4bit-mtp-200k.200k.20261006T030237.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/apple-m5-pro-48gb.qwen3.8-flash-next.slotstream-qwen3.8-flash-next-4bit-32k.32k.20261008T160431.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/apple-m5-pro-48gb.qwen3.8-flash-next.slotstream-qwen3.8-flash-next-4bit-32k.32k.20261008T161700.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/dgx-spark-gb10-128gb.deepseek-v4.1-flash.vllm-deepseek-v4.1-flash-spark-256k-2x.256k.20261003T082323.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/dgx-spark-gb10-128gb.deepseek-v4.1-flash.vllm-deepseek-v4.1-flash-spark-256k-2x.256k.20261003T212655.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/dgx-spark-gb10-128gb.glm-5.3-flash.tensorfold-glm-5.3-flash-exl3-4bpw-1024k-4x.1024k.20261012T165242.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/intel-arc-pro-b70-32gb.nemotron-3.5-bolt-74b-a9b.llama.cpp-sycl-nemotron-3.5-bolt-74b-a9b-24gb-256k-arc-pro-b70.256k.20261017T080702.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/intel-arc-pro-b70-32gb.qwen3.8-flash-next.sglang-qwen3.8-flash-next-exl3-3.05bpw-mtp-nvme12-256k-intel-arc-pro-b70-32gb.256k.20261017T044600.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/intel-arc-pro-b70-32gb.qwen3.8-flash-next.sglang-qwen3.8-flash-next-exl3-3.05bpw-mtp-nvme28-256k-intel-arc-pro-b70-32gb.256k.20261017T024039.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/intel-arc-pro-b70-32gb.qwen3.8-flash-next.sglang-qwen3.8-flash-next-exl3-3.05bpw-mtp-nvme58-256k-intel-arc-pro-b70-32gb.256k.20261017T022820.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/intel-arc-pro-b70-32gb.qwen3.8-flash-next.sglang-qwen3.8-flash-next-exl3-3.05bpw-mtp-nvme58-256k-intel-arc-pro-b70-32gb.256k.20261017T025437.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/intel-arc-pro-b70-32gb.qwen3.8-flash-next.sglang-qwen3.8-flash-next-exl3-3.05bpw-mtp-ram80-256k-intel-arc-pro-b70-32gb.256k.20261016T211050.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/intel-arc-pro-b70-32gb.qwen3.8-flash-next.sglang-qwen3.8-flash-next-exl3-3.05bpw-ram-256k-intel-arc-pro-b70-32gb.256k.20261016T160704.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/radeon-8060s-64gb.qwen3.8-27b.llama.cpp-vulkan-qwen3.8-27b-iq4-xs-mtp-128k.128k.20261020T080217.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3060-12gb.qwen3.6-35b-a3b.llama.cpp-qwen3.6-35b-a3b-mtp-ud-q4-k-xl-autofit-128k-2x.128k.20261001T014311.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3060-12gb.qwen3.8-27b.llama.cpp-qwen3.8-27b-ud-q4-k-xl-tensor-mtp-128k-2x.128k.20261001T013240.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3090-24gb.glm-5.3-flash.exllamav3-glm-5.3-flash-exl3-3.05bpw-nvme-16g-128k-rtx-3090-24gb.128k.20261008T152926.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3090-24gb.glm-5.3-flash.exllamav3-glm-5.3-flash-exl3-3.05bpw-nvme-16g-128k-rtx-3090-24gb.128k.20261009T021643.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3090-24gb.glm-5.3-flash.exllamav3-glm-5.3-flash-exl3-3.05bpw-nvme-16g-128k-rtx-3090-24gb.128k.20261009T124547.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3090-24gb.glm-5.3-flash.exllamav3-glm-5.3-flash-exl3-3.05bpw-nvme-55g-128k-rtx-3090-24gb.128k.20261008T083936.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3090-24gb.glm-5.3-flash.exllamav3-glm-5.3-flash-exl3-3.05bpw-nvme-55g-128k-rtx-3090-24gb.128k.20261009T012633.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3090-24gb.glm-5.3-flash.exllamav3-glm-5.3-flash-exl3-3.05bpw-nvme-55g-128k-rtx-3090-24gb.128k.20261009T123730.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3090-24gb.glm-5.3-flash.exllamav3-glm-5.3-flash-exl3-3.05bpw-offload-128k-rtx-3090-24gb.128k.20261009T064723.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3090-24gb.qwen3.8-27b-pi.sglang-qwen3.8-27b-pi-exl3-3bpw-mtp-vision-200k.200k.20261005T080915.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3090-24gb.qwen3.8-27b.sglang-qwen3.8-27b-exl3-3bpw-mtp-vision-208k.208k.20261010T112034.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3090-24gb.qwen3.8-flash-next.strata-qwen3.8-flash-next-iq3-s-128k-rtx-3090-24gb.128k.20261011T191729.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-3090-24gb.qwen3.8-flash-next.strata-qwen3.8-flash-next-q2-0-128k-rtx-3090-24gb.128k.20261012T015035.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rtx-4080-super-16gb.welp-35b-a3b.llama.cpp-welp-35b-a3b-262k.256k.20261009T194126.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/rx-6700-xt-12gb.qwen3.5-9b.llama.cpp-qwen3.5-9b-q4-k-m-64k-gfx1031.64k.20261011T160703.json` | 1 |
| [local-ai-registry](https://github.com/sybil-solutions/local-ai-registry) | `c0604cff4807` | `lab/runs/x86-64-avx2-cpu.lfm2.5-2.6b.llama.cpp-lfm2.5-2.6b-qad-q4-0-8k-cpu-docker.8k.20261003T102019.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-16g-r16-screen.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-16g-r48-screen.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-55g-pfon-cap8-pair.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-55g-pfon-cap8.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-55g-pfon-n129.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-55g-pfon-nocap-pair.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-58g-pfoff.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-c55-pfoff-control.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-g066a-exact.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-g067-allram.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-image-v4-nvme.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-ab-ctl-16g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-ab-ctl-55g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-ab-k1-16g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-ab-k1-55g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-ab-k1m0-55g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-e2-ctl.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-f-single-16g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-g-cap7-16g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-g-raid-16g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-q-cap7-16g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-q-cap7-55g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-q-raid-16g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-q-raid-55g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-q-single-16g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-q-single-55g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-v44-ctl-16g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-v44-ctl-55g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-v46-16g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n129-v46-full-55g.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n137-b70-55g.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n137-b70-fr120.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n137-b70r-55g.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n137-ctl-55g.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-n137-ctl-fr120.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-s1b-exact-nvme-50g.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-s1c-exact-nvme-55g.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-s1d-exact-nvme-50g.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-s2-exact-nvme-55g.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-s3a-cpu-nvme-55g.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-s3b-cpu-nvme-batched-55g.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-s3b-util-base.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-s3b-util.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-u1-ctrl.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-u1-pflo.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-u1-pfoff.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-u2-lfu.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-u3-vis.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-u5-cpudiag.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-u6-lfu-pfoff.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-u6a-pfoff.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-u6b-pfoff.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-u7-rd32.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-v42-image-smoke-16g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-v42-image-smoke-55g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-v43-image-smoke-16g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/glm53-v43-image-smoke-55g.json` | 1 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/qwen38fn-b70-n130-16g.json` | 10 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/qwen38fn-b70-n130-32g-control.json` | 8 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/qwen38fn-b70-prof-cap7.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/qwen38fn-b70-prof-raid.json` | 2 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/qwen38fn-b70-srv20.json` | 5 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/qwen38fn-b70-srv21.json` | 5 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/qwen38fn-b70-srv22.json` | 5 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/qwen38fn-b70-srv23.json` | 5 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/qwen38fn-b70-srv24b.json` | 3 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/qwen38fn-b70-srv25.json` | 5 |
| [moetier](https://github.com/sybil-solutions/moetier) | `2da3ec482dc9` | `registry/runs/qwen38fn-b70-srv26.json` | 5 |
| [qwen-serve](https://github.com/sybil-solutions/qwen-serve) | `730d8101d0ff` | `qwen3.8-flash-next/b70-offload/results/n130/c32/analysis.json` | 2 |
| [qwen-serve](https://github.com/sybil-solutions/qwen-serve) | `730d8101d0ff` | `qwen3.8-flash-next/b70-offload/results/n130/f16/analysis.json` | 2 |
| [qwen-serve](https://github.com/sybil-solutions/qwen-serve) | `730d8101d0ff` | `qwen3.8/3090-sglang/benchmarks/tp1-results.json` | 4 |
| [qwen-serve](https://github.com/sybil-solutions/qwen-serve) | `730d8101d0ff` | `qwen3.8/3090-sglang/benchmarks/tp2-results.json` | 16 |
| [qwen-serve](https://github.com/sybil-solutions/qwen-serve) | `730d8101d0ff` | `qwen3.8/3090-sglang/benchmarks/tp4-results.json` | 20 |
| [qwen-serve](https://github.com/sybil-solutions/qwen-serve) | `730d8101d0ff` | `qwen3.8/b70/README.md` | 12 |
| [qwen38-flash-next-3090-usb-offload](https://github.com/sybil-solutions/qwen38-flash-next-3090-usb-offload) | `9392e65a9558` | `README.md` | 10 |
| [trellis-serve](https://github.com/sybil-solutions/trellis-serve) | `a1f72be2ff63` | `xpu/README.md` | 4 |
| [trellis-serve](https://github.com/sybil-solutions/trellis-serve) | `a1f72be2ff63` | `xpu/bench/results/2026-09-22.jsonl` | 7 |
| [trellis-serve](https://github.com/sybil-solutions/trellis-serve) | `a1f72be2ff63` | `xpu/bench/results/2026-09-23.jsonl` | 19 |
<!-- END:provenance -->

## License

MIT. The measurements belong to the repos they came from; follow the links.
