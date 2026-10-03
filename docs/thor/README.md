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

The following are source-level tuning candidates; their speed impact has not
been measured on Thor or changed in this iteration:

| Assumption | Affected path | What to measure next |
| --- | --- | --- |
| `kCausalAttentionSmCount = 170` | FP8/INT8/BF16 KV partitioning and prefill tiling | Long-context decode and split/merge traffic with a Thor-sized budget |
| 170-CTA split-K waves | FP8 A8 input, GDN, residual and MLP projection tails | Prefill and partially filled verification tiles; extra reductions may cost more on 20 SMs |
| 99 KiB shared-memory caps | Linear and attention template schedules | Wider tiles and deeper staging within Thor's larger limit; more shared memory can reduce occupancy |
| 170-block RMSNorm and 1,020-block RoPE frontiers | Gating and position encoding | Prefill grid sizes and register occupancy; these ordinary launches are distinct from the fixed cooperative-launch failure |
| Fixed relative GDN recurrence tile costs | Chunked linear-attention prefill | Re-measure tile costs; this path already uses the actual device SM count |
| 5090 format crossovers and Q4 draft-head schedule | Mixed FP8/NVFP4 target and speculative head | Small-batch decode versus prefill under Thor's memory bandwidth and cache limits |

The measured FP4 MLPs already approach the memory bandwidth limit, so tuning
these schedules and increasing accepted tokens per target pass is more promising
than increasing advertised arithmetic utilization alone.

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
