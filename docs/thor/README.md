# Jetson AGX Thor NVFP4 port

This port targets SM110a with CUDA 13.0 and cuBLASLt. It preserves packed NVFP4
weights and uses native FP4 contractions for A4 projections, including fused
SwiGLU, residual, attention-input, and GDN-input projections. Transient storage
belongs to the existing workspace arena. FP8 projections use regular SM110 FP8
MMA. AllowA4 MLP routes switch to native FP4 at two tokens on Thor; single-token
and explicit A16Only routes retain their existing implementations.

BF16, INT8, and FP8 KV storage are supported. NVFP4 and mixed FP8/FP4 KV storage
are rejected at startup. The tested artifact includes text and MTP, without
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

Convert the cached Unsloth checkpoint using the maintained recipe:

```bash
python3 -m tools.convert --model "$CHECKPOINT" \
  --recipe qwen3_8_27b_nvfp4 --source "quantized=$CHECKPOINT" \
  --components text,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal --name qwen3.8-27b-thor \
  --out "$HOME/ninfer-thor/models/qwen3_8_27b_thor.ninfer"
bash tools/thor/run.sh
```

The tested source is `unsloth/Qwen3.8-27B-NVFP4` revision
`f0b7c9e722f5565102fff8481c99e4d86ae099c7`. The recipe preserves its existing
NVFP4 MLP and FP8 projection encodings. Embeddings become FP8; MTP and the
proposal head use the recipe's groupwise encodings. This is a mixed artifact,
with 19.7 GiB of resident weights, rather than an all-FP4 model.

The launch script defaults to MTP4, FP8 KV, eight concurrent request slots,
262,144 total KV tokens, 1,024-token prefill chunks, and automatic restart.
It listens on `127.0.0.1:8000`; use SSH forwarding for remote access.
`NINFER_THOR_DIR` overrides the mounted working directory and
`NINFER_DRAFT_TOKENS` overrides the draft window, within the supported range 1–5.
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

## Verification

All five affected NVFP4 projection suites pass their numerical and CUDA graph
checks. After changing small-batch thresholds, the linear, residual, and SwiGLU
suites pass again, including exact workspace accounting. The causal FP8 attention
suite and the frontend regression suite pass. New frontend checks preserve the
legacy tokenizer's combining-mark boundaries and accept tokenizer.json added
tokens without a redundant config decoder, plus explicit null BOS metadata.

Run the client checks against the local service:

```bash
python3 tools/thor/bench_thor_chat.py --label thor-mtp4 --concurrency 1 --reps 2
python3 tools/thor/bench_thor_chat.py --label thor-mtp4 --concurrency 4 --reps 2
python3 tools/thor/smoke_thor_chat.py
```

These are smoke checks and numerical operator tests, not a broad model quality
evaluation. Published RTX 5090 throughput tables elsewhere in this repository
do not describe Thor performance.
