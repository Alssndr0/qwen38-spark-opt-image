# Decode optimizations for Qwen3.8-Flash-Next on one DGX Spark

A small layer on top of [myllmbox/qwen38-flash-next-recipe](https://github.com/myllmbox/qwen38-flash-next-recipe) and
its image `myllmbox/qwen38-flash-next-vllm:v5.2`: a Dockerfile that builds a v5.2-based image with a few decode speedups,
the scripts to calibrate and check them, and what they measured on one GB10. Everything is opt-in by environment
variable; with none of them set the image serves like v5.2.

Discussed upstream in [myllmbox/qwen38-flash-next-recipe#8](https://github.com/myllmbox/qwen38-flash-next-recipe/issues/8).
The recipe files these numbers come from are in
[Alssndr0/qwen38-flash-next-recipe](https://github.com/Alssndr0/qwen38-flash-next-recipe)
(`recipe.yaml.int8gptq-20261009` for hibrid48, `recipe.yaml.int4-opt-dh-int8-20261009` for INT4-AutoRound).

## Build

```bash
docker pull myllmbox/qwen38-flash-next-vllm:v5.2      # digest sha256:76eda2f6c0e5...
docker build -t qwen38-flash-next-vllm:v5.2-opt6 image/
```

`image/patches/` holds the changes to files that are already in the v5.2 image, as diffs against it; `image/files/`
holds the three new files. A build from this repo is byte-identical, file for file, to the image the numbers below were
measured on (`v5.2-opt6`).

Then in the recipe's `recipe.yaml` set `server.image: qwen38-flash-next-vllm:v5.2-opt6` and add the variables you want
to `env:` (see [Use it](#use-it)).

## What is in it

**Output-exact** (outputs change only within the serve's own run-to-run noise):

| switch | what it does |
|---|---|
| `MBX_SKINNY_GEMM_SM12X: "1"` | Already in v5.2, off there. GB10 plans for the small bf16 decode GEMMs (cuBLAS picks SM80 WMMA kernels on this chip). Not measured on its own here; counted only as part of the set. |
| `OPT_PLE_ASYNC: "1"` | Reads the n-gram (PLE) table rows on a background thread, removing a CPU-GPU sync per forward (0.4 to 10 ms GPU gaps). Byte-checked against the synchronous reader on the NVFP4 and fp8 tables. C library `opt_ple_async.c`. |
| `OPT_MTP_FP8: "1"` | The drafter's 11 projections in FP8 (e4m3 per channel, Marlin W8A16): about 80 MB less read per draft pass. Exact through rejection sampling. |
| `mtp_depth.mode: opt` | Batch draft depth = argmax of expected accepted tokens over measured step time; at 1 stream it keeps the shipped per-request logic. Also always on: the depth bookkeeping no longer syncs GPU tensors to the host. |
| `OPT_DRAFT_HEAD_BASIS: /cache/<basis>.pt`, `OPT_DRAFT_HEAD_SCORER: "fp4"` | Two-stage draft vocabulary head: a rank-256 projection shortlists 2,048 candidates, and only those are scored with the real head. Exact: only the draft proposals can change, and verification against the target is unchanged. Needs a basis fitted per checkpoint. Reads hibrid48's NVFP4 head and INT4-AutoRound's int4 head. |

**Changes outputs slightly** (opt-in):

| switch | what it does |
|---|---|
| `OPT_TARGET_FP8: "1"`, `OPT_TARGET_QFMT: "int8"`, `OPT_TARGET_QGROUP: "128"`, `OPT_TARGET_QCODES: /cache/<codes>.pt` | The target's remaining 16-bit projections (attention, hyper-connections, shared experts, PLE kv, indexer) in INT8 with one scale per 128 weights, through Marlin W8A16, with offline GPTQ codes. Without `OPT_TARGET_QCODES` it rounds to nearest. Without `OPT_TARGET_QFMT` it is FP8 per row, which is the same speed with about 3x the excess divergence over the floor. |

**Always on** (bug fixes, output-exact):

- [vllm#58863](https://github.com/vllm-project/vllm/pull/58863) commit 514102a: the RecoverSSM GDN commit replays the
  accepted tokens with the verify step's own floats, bit-identical to per-token decoding (also offered upstream as an
  optional patch, myllmbox/qwen38-flash-next-recipe#7).
- [vllm#58720](https://github.com/vllm-project/vllm/pull/58720): faster expert-weight loading at boot (85 to 62 s here).
- [vllm#58784](https://github.com/vllm-project/vllm/pull/58784): draft slots that were never proposed are rejected.
  Upstream measured the bug on split prefill/decode setups only, so on one box this is expected to change nothing.

## Measured

One GB10 with the GPU clock capped at 2200 MHz, so absolute numbers are a little lower than an uncapped box. Aggregate
tok/s over 60 s decode windows; single sampled runs vary by about 5 to 10 %. Quality gate: KL and top-1 agreement of
teacher-forced next-token distributions against the stock serve (`scripts/klprobe.py`, about 10,000 positions),
compared with each checkpoint's own run-to-run floor, plus a 400-problem exact-answer eval (`scripts/taskeval.py`).

**hibrid48** (measured on earlier builds of this code, `v5.2-opt3` to `v5.2-opt5`; `v5.2-opt6` adds only the INT4 draft-head path and the state-dump hook):

- Output-exact set vs stock v5.2: +9 to +24 % at 1 stream, +1 to +14 % at 4 streams, +2 to +11 % at 16 streams.
  Prefill unchanged.
- INT8 + GPTQ on top (greedy, fixed depth 7, steps/s): 15.3 to 16.9 at 1 stream (+10 %), 48.0 to 49.4 at 16 (+3 %).

  | target projections | KL | top-1 agreement | task eval (2 runs) |
  |---|---|---|---|
  | bf16 (two captures of the same serve) | 0.016 (floor) | 95.85 % | 347 / 347 |
  | FP8 per row | 0.043 | 93.4 % | 349 / 350 |
  | INT8 per 128, round to nearest | 0.028 | 94.5 % | |
  | **INT8 per 128, GPTQ (12,288 calibration rows)** | **0.0244** | **94.8 %** | **349 / 348** |

  NLL on the probe text is unchanged for all of them. FP8's e4m3 has 3 mantissa bits, so every weight carries about
  2.6 % error whatever the scaling; INT8 per 128 has 0.80 % and runs on Marlin's 8-bit kernel at FP8's speed.
  Quantizing only attention already gives KL 0.022: most of the divergence comes with any 8-bit nudge (it flips a few
  MoE routing decisions), so quantizing less buys little quality for a lot of speed. Group 64 and a second GPTQ pass
  on the quantized model's activations were within noise of the row above.

**INT4-AutoRound** (`v5.2-opt6`):

| prompt | streams | stock v5.2 | output-exact set | + draft head | + INT8/GPTQ for its 16-bit layers |
|---|---|---|---|---|---|
| mixed | 1 | 78.3 | 84.5 | 89.0 | 93.3 |
| mixed | 16 | 332.7 | 359.1 | 374.5 | 377.0 |
| code | 1 | 83.5 | 79.1 | 82.3 | 93.6 |
| code | 16 | 369.6 | 383.9 | 362.9 | 396.2 |
| long prose | 1 | 48.4 | 50.6 | 55.5 | 55.8 |
| thinking on | 1 | 50.4 | 51.4 | 54.0 | 57.4 |

- Single runs per cell. The output-exact set measured +2 to +8 % across the validation suite (one code run at 1 stream
  came in at -5 %).
- Output-exact set: KL 0.058 to 0.061 vs stock, at INT4-AutoRound's own floor (0.059 to 0.065); task eval 328 vs stock
  329 / 333.
- INT8/GPTQ here covers only the layers INT4-AutoRound keeps in 16 bits (hyper-connection mixers, indexer, n-gram
  projection): KL 0.069 against that floor of 0.059 to 0.065.
- In real use, this full INT4 configuration sometimes answered in Chinese. Not investigated, and not seen with
  hibrid48; worth a language check before relying on it.

## Use it

The recipe's `run.sh` mounts its `cache/` directory as `/cache`, so the files below go there.

**hibrid48** (`env:` in `recipe.yaml`, plus `mtp_depth: mode: opt`):

```yaml
  MBX_SKINNY_GEMM_SM12X: "1"
  OPT_PLE_ASYNC: "1"
  OPT_MTP_FP8: "1"
  OPT_DRAFT_HEAD_BASIS: "/cache/draft_basis.pt"
  OPT_DRAFT_HEAD_SCORER: "fp4"
  # optional, changes outputs slightly:
  OPT_TARGET_FP8: "1"
  OPT_TARGET_QFMT: "int8"
  OPT_TARGET_QGROUP: "128"
  OPT_TARGET_QCODES: "/cache/gptq_int8_g128.pt"
```

**INT4-AutoRound**: the same lines, with a basis and codes made on INT4-AutoRound.

### Draft-head basis (per checkpoint)

1. Boot with `OPT_DUMP_DRAFT_H: "/cache/draft_h.pt"` and `OPT_DUMP_ROWS: "8192"`.
   The serve skips the first 2,000 draft calls, then saves up to 4 drafter states per call.
2. Send varied traffic: several languages, code, thinking on. The generation step of `scripts/calib_capture.py` does
   this (it needs `CALIB_OUT` set to any writable directory).
3. Fit: `python3 scripts/fit_draft_basis.py cache/draft_h.pt cache/draft_basis.pt 256`. It prints how much of the
   states' energy rank 256 keeps.

### INT8 + GPTQ codes (per checkpoint)

1. Copy the vLLM source out of the image for the code part of the calibration text:
   `cid=$(docker create qwen38-flash-next-vllm:v5.2-opt6); docker cp $cid:/usr/local/lib/python3.12/dist-packages/vllm vllm-src; docker rm $cid`
2. Boot a calibration serve: the env above without the `OPT_TARGET_*` lines, plus `OPT_CALIB_DIR: "/cache/calib"`,
   `OPT_CALIB_ROWS: "12288"`, `OPT_CALIB_PER_CALL: "300"`, and a smaller KV pool (we used `kv-cache-memory: "16000000000"`).
3. `CALIB_MORE=1 CALIB_OUT=<recipe>/cache/calib CALIB_SRC=vllm-src python3 scripts/calib_capture.py`. It generates text,
   then arms the recorders and prefills it. About 34 GB of recorded rows for hibrid48.
4. Stop the serve, then run GPTQ inside the image (about 3 minutes):
   `docker run --rm --gpus all -v <recipe>/cache:/cache -v $PWD/scripts:/scripts --entrypoint python3 qwen38-flash-next-vllm:v5.2-opt6 /scripts/gptq_int8.py /cache/calib /cache/gptq_int8_g128.pt`
   It reports the held-out output error per layer family for round-to-nearest vs GPTQ.

### Checking a change

- `scripts/klprobe.py capture ref.json` on the stock serve, `capture new.json` on the new one, then
  `scripts/klprobe.py compare ref.json new.json`. The first capture builds the corpus from `KL_SOURCES` (comma-separated
  text files, keep them out of the calibration text). Capture the stock serve twice to get your floor.
- `scripts/taskeval.py LABEL`: 400 generated problems with exact answers, greedy, thinking off.
- `scripts/fixedbench.py LABEL 1,16 768 3`: greedy, `ignore_eos`, fixed prompts, so variants decode nearly the same
  text; prints tok/s and accepted tokens per step. Better than sampled runs for A/B tests.

All scripts talk to `http://127.0.0.1:8000` (`OPT_URL` to change it) and write to `results/` (`OPT_RESULTS`).

## License

Apache-2.0 (see `LICENSE`), the same as vLLM. The diffs in `image/patches/` apply to files in the
`myllmbox/qwen38-flash-next-vllm:v5.2` image.
