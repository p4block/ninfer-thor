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
bash tools/thor/run.sh
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

The launch script defaults to DFlash2 with nine drafts, FP8 KV, eight concurrent request slots,
262,144 total KV tokens, 1,024-token prefill chunks, and automatic restart.
It listens on `127.0.0.1:8000`; use SSH forwarding for remote access.
`NINFER_THOR_DIR` overrides the mounted working directory and
`NINFER_DRAFT_TOKENS` overrides the draft window, within 1–15 for DFlash2.
`NINFER_SPEC_BACKEND=mtp` selects the original MTP artifact and defaults to four
drafts, within 1–5. `NINFER_HOST=0.0.0.0` enables the private LAN listener; this
was explicitly requested for the deployed Thor at `10.69.0.3`.
For the fastest measured single-request mode, use `NINFER_DRAFT_TOKENS=15`;
its four-request explanation throughput is lower than the balanced default.
The original MTP artifact remains available for the MTP backend.
Inspect the service with `docker logs ninfer-thor`; stop it with
`docker stop ninfer-thor`. Recreating it requires removing that stopped container
before running the launch script again.

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
have room: T=64 measured 8.37 ms and roughly 156 GB/s implied traffic. They also
help explain why fifteen drafts do not dominate batched inference.

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
