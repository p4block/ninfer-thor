# Jetson AGX Thor NVFP4 port

This port targets SM110a with CUDA 13.0 and cuBLASLt. It preserves packed NVFP4
weights and uses native FP4 contractions for A4 projections, including fused
SwiGLU, residual, attention-input, and GDN-input projections. Transient storage
belongs to the existing workspace arena. FP8 projections use regular SM110 FP8
MMA. AllowA4 MLP routes switch to native FP4 at two tokens on Thor; single-token
and explicit A16Only routes retain their existing implementations.

BF16, INT8, and FP8 KV storage are supported. NVFP4 and mixed FP8/FP4 KV storage
are rejected at startup. The deployed artifact includes text, MTP, and DFlash2, without
vision. Full 262,144-token context capacity is configured but not validated by
the smoke workload; the tested retrieval prompt contains 9,041 tokens.

## Build and deployment

On the Thor host, build the image with an aarch64 CUDA 13.0 base image:

```bash
docker build -f Dockerfile.thor -t ninfer-thor:sm110 .
```

The tested base image is vLLM commit
`0cbac6cd1305f710e12193596b27488397bcb205`, with CUDA 13.0.88. Its local image ID
was `d49b84dae6751396ebd6b455a2945dcc6c78ca13235b1cdd7c307c21c084491a`.
`BASE_IMAGE` can be overridden with `--build-arg`. The deployed image was
assembled from that existing image plus the same dependencies and verified
binary; the Dockerfile provides the source build equivalent.

Set `CHECKPOINT` to the cached Unsloth source and `DFLASH2_CHECKPOINT` to the
downloaded DFlash2 source described below, then convert with the maintained recipe:

```bash
python3 -m tools.convert --model "$CHECKPOINT" \
  --recipe qwen3_8_27b_nvfp4 --source "quantized=$CHECKPOINT" \
  --source "dflash2=$DFLASH2_CHECKPOINT" \
  --components text,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal --name qwen3.8-27b-thor-dflash2 \
  --out "$HOME/ninfer-thor/models/qwen3_8_27b_thor_dflash2.ninfer"
bash tools/thor/run.sh multi
```

The tested source is `unsloth/Qwen3.8-27B-NVFP4` revision
`f0b7c9e722f5565102fff8481c99e4d86ae099c7`. The recipe preserves its existing
NVFP4 MLP and FP8 projection encodings. Embeddings become FP8; MTP and the
proposal head use the recipe's groupwise encodings. The original MTP artifact
is mixed, with 19.7 GiB of resident weights.

The DFlash2 source is
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`. Set `DFLASH2_CHECKPOINT`
to its downloaded directory containing `config.json` and `model.safetensors`.
The recipe retains the target formats and adds Q8 draft projections and the
BF16 candidate selector. All 659 physical target objects (20.38 GB) were checked
byte for byte against the original artifact; all 963 logical target bindings
retain identical data. The deployed resident weights occupy 21.4 GiB.

Choose the preset on the deployed Thor:

```bash
bash ~/ninfer-thor/run_ninfer_thor.sh single
bash ~/ninfer-thor/run_ninfer_thor.sh multi
```

| Preset | Active request slots | DFlash2 draft tokens | Intended workload |
| --- | ---: | ---: | --- |
| `single` | 1 | 15 | Fastest measured individual decode; additional requests queue |
| `multi` (default) | 8 | 9 | Shared service with a balanced verification window |

Both presets use FP8 KV, 262,144 total KV tokens, 1,024-token prefill chunks,
automatic restart, and the private LAN listener at `0.0.0.0:8000`.
The one-slot preset reserves 9.23 GiB of runtime GPU memory versus approximately
15.2 GiB for eight slots, in addition to the same 21.4 GiB of resident weights.
Its measured startup takes 10.3 seconds. The fresh one-slot baseline reaches
88.53 tok/s for code and 38.23 tok/s for explanation with approximately 166 ms
short-prompt TTFT; this measurement precedes the larger-T head extension.
The OpenAI-compatible base URL is `http://10.69.0.3:8000/v1`.
The script validates the mode, image and artifact, then stops and replaces the
existing container. Switching modes interrupts active requests and reloads the
model; `docker restart` keeps the current preset. `--help` lists the choices.
`NINFER_THOR_DIR` overrides the mounted working directory and
`NINFER_DRAFT_TOKENS` overrides the draft window, within 1–15 for DFlash2.
`NINFER_SPEC_BACKEND=mtp` selects the original MTP artifact and defaults to four
drafts, within 1–5, while retaining the chosen preset's request-slot count.
`NINFER_HOST=127.0.0.1` restricts listening to the Thor itself.
The original MTP artifact remains available for the MTP backend.
Inspect the service with `docker logs ninfer-thor`; stop it with
`docker stop ninfer-thor`.

## Measured results

Measured on a 128 GB Thor in MAXN with clocks locked. The standard-library SSE
client uses two warmed repetitions per prompt, 256 output tokens, temperature
zero, and explicit zero presence/frequency penalties. Decode rate is
`(completion_tokens - 1) / (last_content_time - first_content_time)`; concurrent
aggregate throughput includes the full batch wall time. Speculative streaming
can emit several tokens together, so these are client estimates, rather than
individual token timestamp measurements.

| Configuration | Code decode | Explanation decode | Median short-prompt TTFT |
| --- | ---: | ---: | ---: |
| Initial Thor port, MTP2 | 23.32 tok/s | 21.82 tok/s | 210 ms |
| Native small-batch FP4, MTP2 | 28.87 tok/s | 23.24 tok/s | 174 ms |
| Native small-batch FP4, MTP4 | 34.92 tok/s | 25.57 tok/s | 183–184 ms |

At four concurrent requests, the final MTP4 configuration reaches 98.81 tok/s
aggregate for code and 85.10 tok/s for explanation, including TTFT. Single-request
decode improves about 32% over the initial port using the geometric mean of the
two measured prompt rates. A vLLM rerun stalled during startup and was stopped;
the user's approximate 22 tok/s baseline is contextual evidence, not a matched
engine comparison. An eight-token MTP experiment was rejected by option
validation and did not produce a valid tuning result.

Startup takes approximately ten seconds. The 9,041-token retrieval smoke test
prefills at approximately 1,037 tok/s and returns the correct code in 8.95 seconds.
Arithmetic, Unicode text, and structured tool-call checks also pass. Raw timing
and response records are under [results](results/).

## Thor tuning and bottlenecks

A warmed Nsight Systems public-Engine trace of 64 raw generated tokens with
MTP4 attributes 32.9% of kernel time to native FP4 MLP contractions, 37.0% to
other FP8 projections, 7.3% to the target vocabulary head, and 7.2% to the Q4
draft head. This trace uses the benchmark corpus, rather than the chat prompts;
these percentages characterize that run. Host submission is small compared
with GPU execution; CUDA event synchronization mostly measures waiting for GPU
work.

The main limit is weight-read bandwidth. Native FP4 gate/up and down timings
imply approximately 249 and 238 GB/s respectively, and the largest FP8 GDN
projection approximately 248 GB/s. These are weight bytes divided by measured
kernel time, not memory-counter measurements. Thor's rated bandwidth is
[273 GB/s](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/).
More accepted tokens per verification pass amortize these reads; advertised FP4
compute capacity does not remove this small-batch memory limit.

The vocabulary head now uses fewer K-reduction warps on Thor while retaining
BF16 activations and FP32 accumulation. Its public Linear latency at T=5 falls
from 7.158 ms to 4.918 ms, a 31.3% reduction. Across T=1–41, the final curve's
largest latency improvement is 32.4%; its worst measured increase is 0.92% at
T=29. The first candidate regressed up to 2.8% in the T=25–32 interval; the final
implementation retains the original schedule there. Larger-T head GEMMs still
have room: T=64 measured 8.37 ms and roughly 156 GB/s implied traffic.

The next head comparison compiled a single candidate-by-extent matrix at twelve
extents within T=42–96 and qualified every eligible candidate against independently decoded
FP8 weights and an FP64 dot-product oracle before timing. Two K warps win at
T=42–56 and T=65–96. The existing GEMM remains at T=57–64: the sliced-K
64-token two-warp alternative regresses by up to 2.9% there. Across the whole
matrix, a padded 96-token/four-warp candidate was 52.3% slower at T=64 and was
rejected. The selected routes retain BF16
activations, FP32 accumulation and zero caller workspace; no weights change.
Raw qualification and timings are in [results](results/head-wide-sweep-qualified.csv).
The final public Linear curve improves latency by 8.6–20.8% at T=42–56 and
5.1–26.9% at T=65–96. T=80 falls from 14.00 ms to 10.48 ms. The unchanged
T=57–64 interval measures between a 0.19% improvement and a 0.37% increase;
these small changes are retained in the results, rather than classified as a
demonstrated speedup. The public FP8 A16 oracle suite passes the new boundaries,
larger calls with sliced tails, output guards and graph replay. The real DFlash2
Engine test also passes at K=9 with eight active request slots, CUDA graphs,
sampling, context restore and no draft-state host transfers.

The deployed extension was compared with a fresh same-preset baseline using the
same two warmed repetitions, 256 output tokens and code/explanation prompts:

| Active requests | Code aggregate before → after | Explanation aggregate before → after |
| --- | ---: | ---: |
| 4 | 137.54 → 142.83 tok/s | 93.02 → 97.71 tok/s |
| 8 | 147.37 → 148.91 tok/s | 106.99 → 108.08 tok/s |

At eight requests, aggregate rates increase 1.05% and 1.02%, while per-request
median decode rises from 25.22 to 25.58 and 18.21 to 18.46 tok/s. At four
requests, higher aggregate rates accompany slightly lower per-request median
decode: 45.21 to 45.03 (−0.39%) and 27.69 to 27.62 (−0.27%) tok/s.
Four-request median TTFT increases from 699 to 769 ms for code and 702 to 719 ms
for explanation. Report both metrics; aggregate rates alone do not describe
individual response latency.

Greedy responses changed in 8/16 eight-request code samples and 12/16 explanation
samples, and in 2/8 and 4/8 four-request samples. The target weights are unchanged,
but floating-point reduction schedules and live batching can change token
decisions. These small, content-dependent end-to-end gains are workload results,
not evidence of bitwise identity or a broad quality evaluation. The independent
numerical criteria, Engine checks and four API smoke cases pass. Full 262K
context and RTX 5090 execution remain unverified for this extension.

The cooperative BF16 GDN gating launcher previously assumed SM120 residency.
A standalone large-prefill integration check aborted with
`cudaErrorCooperativeLaunchTooLarge`. Thor now queries occupancy for the actual
compiled full and predicated kernels and slices the token grid within the safe
resident limit. The public oracle suite and the previously failing integration
checks pass after the fix. During earlier parallel diagnostics with three model
instances, the NVIDIA driver hung and Thor was rebooted at the user's request;
subsequent GPU checks run one model instance at a time. The debugger established
the cooperative launch failure, but does not establish the driver's internal
hang mechanism.

| Configuration | C1 code decode | C1 explanation decode | C4 code aggregate | C4 explanation aggregate |
| --- | ---: | ---: | ---: | ---: |
| Resumed MTP4 baseline | 34.85 | 25.53 | 98.81 | 85.10 |
| Head tuned, MTP4 | 35.64 | 26.17 | — | — |
| Head tuned, DFlash2 K=7 | 44.99 | 28.52 | 126.66 | 95.34 |
| Deployed head/GDN fixes, DFlash2 K=9 | 46.45 | 28.74 | 137.53 | 92.99 |
| Head tuned, DFlash2 K=15 | 86.62 | 37.41 | 125.57 | 73.06 |

All rates are tokens/s with the client methodology above. The C4 MTP baseline
reuses the prior same-clock measurements; the C1 MTP baseline was refreshed.
Each candidate uses two warmed repetitions and 256 requested output tokens.
The nine-draft preset improves all four rates over MTP4 and is the balanced
default. Fifteen drafts wins for one request but slows C4 explanation by 14.1%
relative to MTP4 and 21.4% relative to nine drafts. Nine drafts trades 2.5% lower
C4 explanation throughput for 8.6% higher C4 code throughput compared with seven.

Telemetry on the C1 code prompt shows about 3.59 output tokens per target pass
with MTP4, 4.47 with DFlash2 K=7, and 9.11 with K=15. Wider blocks also cost more
per pass and do not retain this acceptance advantage on every workload.
Greedy texts differ across draft widths, even though target tensor bytes are
identical; batch-dependent numerical paths can change decisions. These are
throughput and correctness smoke measurements, not proof of bitwise sequence
identity or a broad model-quality evaluation.

DFlash2 reserves approximately 5 GiB more GPU memory than MTP4, including its
extra weights and runtime reservation. K=7 startup measured 16 seconds versus
10 seconds for MTP4. Its 9,041-token retrieval smoke took 9.23 seconds versus
8.95 seconds for the earlier MTP4 run, a 3.1% increase; both returned the code.
The packaged nine-draft deployment repeats C1 and C4 timings after the GDN fix,
with 163 ms C1 and approximately 700 ms C4 short-prompt TTFT. Its retrieval smoke
takes 9.13 seconds, a 2.0% increase over the earlier MTP4 measurement. Arithmetic,
Unicode and structured tool calls also pass. Both nine- and fifteen-draft
real-model integration tests pass with CUDA graphs and four active requests,
covering sampling, forced thinking, page settlement and context restore.
The wider block's smoke and integration checks also pass. Full 262K context,
vision, and a broad reasoning workload remain unverified.

The Linear benchmark now uses the Thor bandwidth reference and leaves
uncalibrated sustained-read and dense-compute references unavailable instead
of reporting RTX 5090 values. The saved baseline head CSV predates that metadata
fix; compare its timings, rather than its old reference percentages. A stale
tokenizer test executable was relinked, and one oversized shared-memory sweep
candidate was excluded before the final qualification; neither was deployed.
The changed GPU routes were verified on Thor; RTX 5090 tests were not rerun.

## Remaining 5090 assumptions

CUDA device properties on this Thor report 20 SMs, 32 MiB of L2, 228 KiB of shared
memory per SM, a 227 KiB opt-in per-block limit, and 1,536 threads per SM.
The [RTX 5090 architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
specifies 170 SMs, 96 MiB of L2 and 1,792 GB/s memory bandwidth. These differences
affect launch schedules and operand reuse even when both chips support FP4.

The current scheduling iteration evaluates attention partition budgets, FP8 A8
split-K waves and shared-memory schedules, paired RoPE, gated RMSNorm and GDN
recurrence tiles. The workload priority is agent sessions around **50K tokens**;
100K and 150K are secondary. The 262K launch limit remains a capacity setting,
not a validated recall or performance claim. No recall-quality benchmark has
established a safe context cutoff.

| Path | Selected policy or disposition |
| --- | --- |
| FP8 KV attention | Thor-sized grouped partition budgets by heads, batch and verification width; tiled prefill uses a 20-CTA wave. Short single-request caches retain the original budget. INT8/BF16 KV partitioning is not retuned. |
| FP8 attention input | Bulk split-K wave reduced from 170 to 20; workspace reservation starts at T=289 and matches the actual schedule. |
| FP8 residual projections | Remove split-K from medium schedules for K=6144 and K=17408; retain small and large schedules. |
| Paired text and DFlash2 RoPE | Thor-specific block sizes by token extent; other position-encoding routes keep their existing dispatch. |
| Larger shared-memory FP8 tiles | Rejected: the candidate matrix includes slowdowns up to 381.9%; retain the existing caps. |
| Gated RMSNorm | Retain the current prefetch frontier: unprefetched small extents are 26–28% slower. |
| GDN recurrence | Retain DV64 for the active 48-head model: smaller tiles are up to 42.8% slower at T=1024. |
| Other format crossovers and Q4 draft head | Not retuned in this iteration. |

The candidate matrices are finite task-local experiments, qualified before timing.
Packed FP8 operands are independently decoded; attention and FP8 dot products
use FP64 references, RoPE uses independent FP64 trigonometry, and GDN checks
output and final state against its independent oracle. Numerical criteria match
the public suites; no thresholds were relaxed to qualify candidates. Matrix
inputs cover both registered attention geometries where applicable, the active
27B GDN geometry, verification extents and 1,024-token prefill.

### Scheduling measurements

Baseline is commit `27048cd`, using the same artifact, MAXN, fixed CPU 2601 MHz,
GPU 1575 MHz and EMC 4266 MHz. CUDA is 13.0.88 and the driver is 595.78.
Initial candidate timings were taken while CPU/GPU clocks still scaled despite
EMC being fixed. Those records remain in the working results as
`dynamic-public-*`; published before/after public Op and Engine results were
repeated after verifying fixed clocks. The launcher now runs `jetson_clocks`
automatically; `NINFER_LOCK_CLOCKS=0` opts out. This uses a CPU-only privileged
maintenance container, followed by the ordinary inference container.

| Public route / workload | Result versus baseline |
| --- | --- |
| FP8 append attention, 72 batch/width/context extents | Median latency −7.83%; 45 improve >2%, 3 regress >2%; range −35.13% to +10.66%. |
| Active 24-head attention, longer single-request caches | Measured latency reductions approximately 5.6–22.5% at 2K–32K. |
| FP8 tiled prefill with 8K cached context | 24-head T=128 −10.75%, T=1024 −7.8%; 16-head T=128 −13.95%, T=1024 −0.29%. |
| Fused FP8 attention input | T=289–1024 reduces latency by 0.14–4.39%; T=256/288 essentially unchanged. |
| Plain FP8 attention-input geometry | T=1024 −4.78%; T=896 **+2.71%**, retained as a measured tradeoff. |
| Fused FP8 residual K=6144 | T=256 −14.27%, T=768 −6.65%; other measured extents within +0.3%. |
| Fused FP8 residual K=17408 | T=192/256/384 −13.49/−10.29/−10.82%; other extents within +0.3%. |
| Paired RoPE, 35 public extents | Median −11.0%, range −35.47% to **+14.19%**. |

Attention/projection measurements use public Ops, CUDA graphs, warmup four and
16 measured repetitions; attention/input use cold 256 MiB cache flushes. RoPE
uses the public benchmark's warm eager timings. Its saved logs precede the
bandwidth-reference metadata fix; compare microseconds, not their old 5090
roofline percentages.

Adverse measurements remain visible: secondary 16-head B1/context-256 attention
is +10.66% at width one, +3.10% at width ten and +2.58% at width sixteen even
with the original partition budget restored. DFlash2 RoPE T=7 is +14.19%
(approximately 4.51→5.15 µs), although that split-five dispatch is unchanged.
Their cause is unresolved; these observations are not dismissed as noise.
Three alternating baseline/selected repeats give zero median change for the
secondary width-ten/sixteen cases and DFlash2 T=16, and +0.67% for DFlash2 T=7
(the baseline repeated medians span 4.45–5.63 µs). The secondary width-one
attention difference persists at +8.57%; its cause remains unresolved. The
original adverse measurements are preserved rather than replaced.
Attention-input workspace at T=1024 drops from 27,529,216 to 7,868,416 bytes;
T=289/384 adds partial storage and a reduction where the baseline was unsplit.

| Public Engine workload, two warmed repetitions | Baseline | Tuned |
| --- | ---: | ---: |
| Single, short code, decode | 88.62 tok/s | 88.51 tok/s |
| Single, short explanation, decode | 38.24 tok/s | 38.23 tok/s |
| Four requests, short code, aggregate | 137.63 tok/s | 137.65 tok/s |
| Four requests, short explanation, aggregate | 93.09 tok/s | 93.02 tok/s |
| Single, 8,674-token cached context, decode | 32.51 tok/s | 38.78 tok/s |
| Same cached-context request, total duration | 8.029 s | 6.759 s |

Short workloads change by less than 0.2%; the 8.7K-context rate improves 19.3%
and total duration falls 15.8%. These client estimates use 256 output tokens,
greedy sampling and disabled thinking. Both long-context outputs and four of
eight concurrent explanation outputs differ; independent Op qualification and
smoke checks establish the tested correctness scope, not sequence identity.
The long-context prompt is prefix-cached, so its approximately 182 ms TTFT does
not measure cold prefill. Weight bandwidth still limits short requests; deeper
attention and prefill require separate measurements.

The subsequent deep-context comparison uses exactly **50,000 prompt tokens**,
a synthetic agent tool-history prompt calibrated with the server tokenizer,
and the same 256-token greedy request on both builds. One cold request and two
cached continuations were measured per build. Cached median decode rises from
24.21 to 28.25 tok/s (**+16.7%**); cached request duration falls from 10.762 to
9.249 seconds. Cold TTFT falls from 59.393 to 58.607 seconds (**−1.3%**), so cold
prefill remains the largest waiting cost. This narrow synthetic workload does
not establish performance or recall on real agent trajectories. Generated
texts differ between builds, as recorded in the raw JSONL evidence.

A further independently qualified attention matrix covers active 24-head
geometry at 32K, 50K, 100K and 150K, batches one/four and widths one/ten/sixteen.
All 96 candidates pass the FP64 reference before timing at fixed clocks. At
50K, the selected single-request policy reduces attention latency by 10.86%
(width one), 3.05% (width ten) and 2.64% (width sixteen). The selected four-request
decode policy gains 7.22%; four-request verification retains the old budget
because the smaller alternatives regress about 24%. At 100K/width sixteen the
chosen policy is 0.44% slower. These deep-context results support retaining the
coherent current dispatch rather than choosing individual best-case anchors.

The deep prefill matrix also qualifies all 16 candidates at T=1024 and
32K/50K/100K/150K cached depths. The selected tiled policy reduces latency by
2.93/2.20/1.66/1.21% respectively and workspace from 177,537,024 to
126,812,160 bytes. Increasing its wave to 40 or 80 yields the same partition
schedule and no material further gain. This supports keeping the 20-CTA policy;
it does not remove the roughly minute-long cold 50K prefill cost.

At 50K, draft-window testing (5/9/15) gives cached decode medians
**30.34/25.92/28.25 tok/s** on the synthetic history. Five drafts gain another
7.37% over fifteen, but short code falls to 43.40 tok/s and short explanation to
28.80 tok/s, versus 88.51/38.23 with fifteen. The output sequences differ, so
these are workload measurements, not pure kernel speedups. Fifteen remains the
single-user default; use the existing override for the measured deep workload:

```bash
NINFER_DRAFT_TOKENS=5 bash ~/ninfer-thor/run_ninfer_thor.sh single
# Return to the default single-user window:
bash ~/ninfer-thor/run_ninfer_thor.sh single
```

A measured-range Nsight Systems profile of the public Engine at 50K input plus
64 generated tokens attributes **51.25% of GPU kernel time to FP8 projections,
27.01% to attention, 11.14% to native FP4 contraction/finish work, 2.91% to GDN,
and 7.69% to other kernels**. It uses the token-ID benchmark corpus, five drafts,
FP8 KV, 1,024-token prefill chunks, fixed clocks, no context retention and no
warmup; load and decode-graph priming are outside the capture. It measures cold
prefill plus decode, rather than the synthetic agent-history chat. Prefill is
58.536 seconds (854 tok/s), decode 1.220 seconds. The total 58.487 seconds of
summed kernel time identifies bulk FP8 projections and tiled attention as the
largest remaining cold-context costs; it does not establish hardware-counter
bandwidth utilization. The five-draft real-model graph integration also passes.

An extended RMSNorm stress fixture at T=1024 failed the existing gross-error
bound: maximum error 0.05703 against bound 0.04500. Both candidate schedules
and the unchanged public Op are bit-identical, and the nearest BF16 cast of the
FP64 result has that same error. No criterion was relaxed and no RMSNorm change
was shipped. The initial private attention harness also hit a CUDA registration
collision; unique private launch names fixed it before qualification/timing.
The affected public FP8 attention, RoPE, FP8 A8 Linear, LinearAdd and complete
attention-input suites pass, including workspace intervals and CUDA graphs.
Real DFlash2 integration passes for both presets; arithmetic, Unicode, tool
calls and 9K retrieval smoke checks pass. RTX 5090 and vision were not rerun.
Python checks used verified local 3.14 and Thor 3.12; the specified maintainer
Python 3.11 executable is absent on this workstation.

### BetterBench reports in tmux

[BetterBench](https://github.com/GGZ14/BetterBench) is installed in
`~/ninfer-thor/betterbench-venv` from revision `d00ad5e`. The wrapper adds the
explicit NInfer `enable_thinking=false` request option and calibrates synthesized
prefill depths with the server tokenizer before each timed request; it preserves
BetterBench's timers and metrics. Configuration uses greedy sampling, seed 1234, three warmups,
20 passes per category, and concurrency 1/2/4/8 with 48 requests per level.
The prefill sweep requests depths 2K, 8K, 32K, 50K, 64K, 100K and 150K; these
are calibrated synthesis targets, and reports use the server's actual token counts.
The extra `agentic50k` report uses a tokenizer-calibrated 50K synthetic agent
history with prefix reuse. It evaluates continuation speed, not task success
or recall. Standard prefill uses fresh nonces and randomized bodies to avoid
prefix-cache hits.

```bash
# Installed host paths; launch after inference tuning has finished.
revision=$(git rev-parse HEAD)
tmux new-session -d -s ninfer-betterbench \
  "NINFER_REVISION=$revision bash $HOME/ninfer-thor/betterbench-tools/run_betterbench.sh"
tmux attach -t ninfer-betterbench
```

The runner switches to `single` for decode and cached 50K continuation reports,
compares five drafts on the same 50K history, then switches to `multi` for
decode, cold prefill and concurrency. Switching reloads the
model and interrupts other requests; it keeps one resident model and leaves
multi mode running. Reports are saved under
`~/ninfer-thor/reports/betterbench-<UTC timestamp>/` as offline HTML, JSON and
logs, with configuration, BetterBench revision and image ID alongside them.
Do not run competing GPU benchmarks during this job. The deep report is
included when `~/ninfer-thor/deep-context-50k-prompt.json` from the tuning run
exists. A full report is substantially longer than a `--quick` smoke run;
quick reports are not publishable throughput evidence.

## All-layer NVFP4 experiment

`qwen3_8_27b_nvfp4_all_layers` converts all large projections in the 64 target
transformer layers to NVFP4 with `AllowA4` activations. The original first-56-layer
NVFP4 MLPs remain byte-identical; 344 formerly FP8 logical projection bindings
are requantized from their represented checkpoint values. The resulting target
uses 256 NVFP4 physical projection parents. FP8 remains for the embedding and
vocabulary head; BF16/FP32 remains for norms, convolutions and recurrent
controls. MTP, DFlash2 and the Q4 proposal head retain their original formats.
This is not an all-tensor or FP4-KV conversion.

The new offline `nvfp4_block_maxabs` method uses K16 E4M3FN block scales, E2M1
codes and one FP32 divisor per fused parent. It rounds scales before codes with
nearest/ties-to-even rounding. New activation uses have divisor one and dynamic
block scaling, without activation-aware calibration. The existing mixed recipe
remains available; the experiment writes a separate artifact.

Reproduce conversion from the same local checkpoint and companion paths:

```bash
python3 -m tools.convert --model "$CHECKPOINT" \
  --recipe qwen3_8_27b_nvfp4_all_layers --source "quantized=$CHECKPOINT" \
  --source "dflash2=$DFLASH2_CHECKPOINT" --components text,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal --device cpu --name qwen3.8-27b-thor-all-layers-nvfp4 \
  --out "$HOME/ninfer-thor/models/qwen3_8_27b_thor_all_layers_nvfp4.ninfer"
```

The artifact shrinks from 23,431,064,580 to 19,335,861,252 bytes (17.5%). A byte
audit checks all 718 unchanged logical weight bindings against the mixed
artifact; none differ, including vocabulary, original NVFP4 MLPs and companion
weights. Conversion took 525 seconds on Thor with four CPU threads and did not
use GPU inference or GPU quantization.

A fixed quality comparison scores the first 4,096 Unicode characters of one
stream from each domain in `ninfer-ppl-1m-v1`, with separate root histories,
context 2,048, stride 1,024 and FP8 KV. Both use public `ninfer-perplexity` with
the same tokenizer and current binary. This covers 6,460 scored tokens and is
a limited quantization check, not a broad reasoning or deep-recall evaluation.

| Domain | Tokens | Mixed PPL | All-layer NVFP4 PPL |
| --- | ---: | ---: | ---: |
| Chinese reference | 2,930 | 12.206824 | 12.535150 |
| English long form | 1,080 | 5.681942 | 5.915276 |
| English reference | 1,000 | 4.942526 | 5.192955 |
| NInfer code | 1,450 | 1.710424 | 1.751130 |
| Token-weighted overall | 6,460 | 6.007907 | 6.201389 |

Overall PPL rises 3.22%, with mean NLL 1.793076→1.824773; every domain degrades.
Do not interpret faster inference as equivalent model quality. The baseline
quality run overlaps CPU conversion, so its scoring wall rate is not a valid
matched speed comparison.

CPU rounding/packing, global-divisor streaming and conversion checks pass (14
checks on verified Thor Python 3.12); independent public FP64 projection and
state suites pass for Linear A4, attention input, GDN input/snapshot/record,
residual and SwiGLU. The initial snapshot executable threw a cuBLASLt selection
error; relinking it makes two subsequent full runs pass. The precise cause is
not established. The first conversion was interrupted to fix missing automatic
fused-parent packing for the new built-in method; it was not used for inference.
The five-draft real-model graph integration passes. A corpus-manifest setup
error and fixture binding-lookup errors were corrected before the comparisons.
Local checks use verified Python 3.14 for compilation only; local PyTorch and
the prescribed Python 3.11 interpreter are unavailable.

The matched public-serving comparison uses the same `ninfer-serve` binary,
artifact tokenizer/template, fixed MAXN clocks, FP8 KV, one request slot, five
drafts, greedy sampling and 256 generated tokens. Each artifact starts from a
fresh server. The 50K history is the same synthetic agent log used in scheduling
tuning; explanation has exactly 50,000 input tokens and code 49,996. One cold
explanation and two cached continuations per prompt are measured. Code is first
primed with 32 generated tokens; that changed-query request misses the prefix
cache on both artifacts and reloads the history. These are client SSE estimates,
not individual token timestamps or agent-task success measurements.

| Workload, five drafts | Mixed | All-layer NVFP4 | Change |
| --- | ---: | ---: | ---: |
| Cold 50K explanation TTFT | 58.603 s | 33.925 s | −42.11% |
| Same cold request, total duration | 67.000 s | 42.136 s | −37.11% |
| Cached 50K explanation decode | 30.36 tok/s | 31.06 tok/s | +2.31% |
| Cached 50K code decode | 47.01 tok/s | 47.67 tok/s | +1.41% |
| Short code decode | 43.43 tok/s | 45.65 tok/s | +5.12% |
| Short explanation decode | 28.82 tok/s | 34.65 tok/s | +20.25% |

Short cached TTFT falls from approximately 169 to 143 ms; 50K cached TTFT falls
from 224–227 to 197–199 ms. All compared greedy output sequences differ. On the
50K explanation, draft acceptance falls from 179/375 (47.7%) to 172/407 (42.3%),
offsetting much of the faster target execution. This is a substantial cold
prefill gain and a small cached-context decode gain at the measured scope.
Arithmetic, Unicode, structured tool-call and 9K retrieval checks all pass;
retrieval duration drops from approximately 8.9 to 4.5 seconds. The nine-draft,
eight-request real-model integration also passes with CUDA graphs and state
restore. No full 100K/150K Engine comparison or broad recall/reasoning test is
established by these results.

With the all-layer artifact, widening from five to fifteen drafts helps some
workloads and hurts others under the same serving conditions:

| Workload | Five drafts | Fifteen drafts | Change |
| --- | ---: | ---: | ---: |
| Cached 50K explanation | 31.06 tok/s | 25.55 tok/s | −17.75% |
| Cached 50K code | 47.67 tok/s | 57.17 tok/s | +19.92% |
| Short code | 45.65 tok/s | 52.78 tok/s | +15.61% |
| Short explanation | 34.65 tok/s | 38.08 tok/s | +9.91% |

The 50K explanation accepts 172/1,193 drafts (14.4%) with the wider window,
versus 172/407 (42.3%) with five. The fifteen-draft graph/state integration
passes, but widening is not a universal decode improvement. Five remains the
recommendation for this measured deep-context explanation workload. These
window results compare the experimental artifact to itself, not a freshly
repeated mixed-model fifteen-draft run.

A separate public `ninfer_bench` Nsight Systems capture uses the same 50K
input-token fixture, 64 output tokens, five drafts, FP8 KV, 1,024-token prefill
chunks and fixed clocks. Captured kernel time falls from 58.49 to 34.11 seconds;
prefill falls from 58.54 to 34.16 seconds. The FP8-KV attention kernel alone now
accounts for 46.3% of kernel time, versus approximately 27.0% before. Remaining
cost includes NVFP4 contractions, separate BF16 finish/epilogue kernels,
activation quantization, GDN and convolution. FP8 vocabulary-head work remains.
This is a cold-prefill-plus-decode trace, not a decode-only bandwidth-counter
measurement. It supports targeting attention and speculation acceptance next;
it does not establish achieved memory bandwidth or agent-task quality.

Launch the experimental artifact explicitly:

```bash
NINFER_ARTIFACT=/work/models/qwen3_8_27b_thor_all_layers_nvfp4.ninfer \
  NINFER_DRAFT_TOKENS=5 bash ~/ninfer-thor/run_ninfer_thor.sh single
# Restore the mixed model with the usual preset:
bash ~/ninfer-thor/run_ninfer_thor.sh single
```

`NINFER_ARTIFACT` chooses an explicit container path under `/work`; preset,
backend, KV storage and other launch choices still apply. The unfinished
BetterBench session was interrupted for this sequential experiment, with its
logs retained. The `ninfer-betterbench` tmux session queues full mixed-model
then experimental-artifact runs sequentially, with reports under
`~/ninfer-thor/reports/betterbench-20261003-nvfp4/{mixed,nvfp4}/`.
Each runs short-category decode, cached 50K decode with fifteen and five drafts,
then multi-user decode, cold prefill through 150K and concurrency. The queue
restores the mixed multi-user preset after both runs. Reports are asynchronous;
these tables do not claim completed BetterBench results.

```bash
tmux attach -t ninfer-betterbench
# Completed runs write single.html, agentic50k.html, agentic50k-k5.html and multi.html.
```

## Verification

All five affected NVFP4 projection suites pass their numerical and CUDA graph
checks. After changing small-batch thresholds, the linear, residual, and SwiGLU
suites pass again, including exact workspace accounting. The causal FP8 attention
suite and the frontend regression suite pass. New frontend checks preserve the
legacy tokenizer's combining-mark boundaries and accept tokenizer.json added
tokens without a redundant config decoder, plus explicit null BOS metadata.

Run the client checks against the local service:

```bash
python3 tools/thor/bench_thor_chat.py --label thor-dflash2-k9 --concurrency 1 --reps 2
python3 tools/thor/bench_thor_chat.py --label thor-dflash2-k9 --concurrency 4 --reps 2
python3 tools/thor/smoke_thor_chat.py
```

These are smoke checks and numerical operator tests, not a broad model quality
evaluation. Published RTX 5090 throughput tables elsewhere in this repository
do not describe Thor performance.
