#!/usr/bin/env bash
set -euo pipefail
usage() {
  cat <<'EOF'
Usage: run.sh [single|multi]
  single  One active request; DFlash2 with 15 draft tokens.
  multi   Eight active requests; DFlash2 with 9 draft tokens (default).

Starts or replaces the ninfer-thor container, listening on 0.0.0.0:8000.
Overrides: NINFER_THOR_DIR, NINFER_HOST, NINFER_DRAFT_TOKENS,
           NINFER_SPEC_BACKEND (dflash2 or mtp; MTP defaults to 4 drafts),
           NINFER_LOCK_CLOCKS=0 (leave existing clock policy; default locks clocks).
EOF
}
if [[ $# -gt 1 ]]; then usage >&2; exit 2; fi
mode="${1:-multi}"
case "$mode" in
  single) max_concurrency=1; default_draft_tokens=15 ;;
  multi) max_concurrency=8; default_draft_tokens=9 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
ninfer_dir="${NINFER_THOR_DIR:-$HOME/ninfer-thor}"
spec_backend="${NINFER_SPEC_BACKEND:-dflash2}"
case "$spec_backend" in
  dflash2)
    artifact="/work/models/qwen3_8_27b_thor_dflash2.ninfer"
    ;;
  mtp)
    artifact="/work/models/qwen3_8_27b_thor.ninfer"
    default_draft_tokens=4
    ;;
  *)
    printf 'Unsupported Thor speculative backend: %s\n' "$spec_backend" >&2
    exit 2
    ;;
esac
draft_tokens="${NINFER_DRAFT_TOKENS:-$default_draft_tokens}"
max_draft_tokens=15
if [[ "$spec_backend" == mtp ]]; then max_draft_tokens=5; fi
if [[ ! "$draft_tokens" =~ ^([1-9]|1[0-5])$ ]] || (( draft_tokens > max_draft_tokens )); then
  printf 'Draft tokens must be in 1–%s for %s.\n' "$max_draft_tokens" "$spec_backend" >&2
  exit 2
fi
listen_host="${NINFER_HOST:-0.0.0.0}"
# Validate the preset, image and artifact before replacing a running service.
docker image inspect ninfer-thor:sm110 >/dev/null
if [[ ! -f "$ninfer_dir${artifact#/work}" ]]; then
  printf 'Missing model artifact: %s\n' "$ninfer_dir${artifact#/work}" >&2
  exit 2
fi
# Clock scaling makes short-kernel timings and request latency inconsistent.
# Use the CPU-only runtime so this maintenance container does not open CUDA.
if [[ "${NINFER_LOCK_CLOCKS:-1}" == 1 ]]; then
  docker run --rm --runtime runc -e NVIDIA_VISIBLE_DEVICES=void \
    --privileged --pid host --entrypoint nsenter ninfer-thor:sm110 \
    -t 1 -m -p /usr/bin/jetson_clocks
fi
if docker container inspect ninfer-thor >/dev/null 2>&1; then
  docker stop --timeout 30 ninfer-thor >/dev/null
  docker rm ninfer-thor >/dev/null
fi
printf 'Launching %s: %s, %s drafts, %s active request slots at %s:8000\n' \
  "$mode" "$spec_backend" "$draft_tokens" "$max_concurrency" "$listen_host"
docker run -d --name ninfer-thor --restart unless-stopped \
  --init --runtime nvidia --network host --ipc host \
  -v "$ninfer_dir:/work" \
  --entrypoint /opt/ninfer/build/apps/ninfer-serve ninfer-thor:sm110 \
  "$artifact" \
  --host "$listen_host" --port 8000 \
  --model-id unsloth/Qwen3.8-27B-NVFP4 \
  --max-context 262144 --kv-capacity 262144 --max-concurrency "$max_concurrency" \
  --kv-dtype fp8 --prefill-chunk 1024 --spec "$spec_backend" \
  --draft-tokens "$draft_tokens" --lm-head-draft \
  --request-log-jsonl /work/ninfer-requests.jsonl
