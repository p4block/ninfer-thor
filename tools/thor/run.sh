#!/usr/bin/env bash
set -euo pipefail
ninfer_dir="${NINFER_THOR_DIR:-$HOME/ninfer-thor}"
spec_backend="${NINFER_SPEC_BACKEND:-dflash2}"
case "$spec_backend" in
  dflash2)
    artifact="/work/models/qwen3_8_27b_thor_dflash2.ninfer"
    default_draft_tokens=9
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
listen_host="${NINFER_HOST:-127.0.0.1}"
docker run -d --name ninfer-thor --restart unless-stopped \
  --init --runtime nvidia --network host --ipc host \
  -v "$ninfer_dir:/work" \
  --entrypoint /opt/ninfer/build/apps/ninfer-serve ninfer-thor:sm110 \
  "$artifact" \
  --host "$listen_host" --port 8000 \
  --model-id unsloth/Qwen3.8-27B-NVFP4 \
  --max-context 262144 --kv-capacity 262144 --max-concurrency 8 \
  --kv-dtype fp8 --prefill-chunk 1024 --spec "$spec_backend" \
  --draft-tokens "$draft_tokens" --lm-head-draft \
  --request-log-jsonl /work/ninfer-requests.jsonl
