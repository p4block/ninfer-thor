#!/usr/bin/env bash
set -euo pipefail
ninfer_dir="${NINFER_THOR_DIR:-$HOME/ninfer-thor}"
draft_tokens="${NINFER_DRAFT_TOKENS:-4}"
docker run -d --name ninfer-thor --restart unless-stopped \
  --init --runtime nvidia --network host --ipc host \
  -v "$ninfer_dir:/work" \
  --entrypoint /opt/ninfer/build/apps/ninfer-serve ninfer-thor:sm110 \
  /work/models/qwen3_8_27b_thor.ninfer \
  --host 127.0.0.1 --port 8000 \
  --model-id unsloth/Qwen3.8-27B-NVFP4 \
  --max-context 262144 --kv-capacity 262144 --max-concurrency 8 \
  --kv-dtype fp8 --prefill-chunk 1024 --spec mtp \
  --draft-tokens "$draft_tokens" --lm-head-draft \
  --request-log-jsonl /work/ninfer-requests.jsonl
