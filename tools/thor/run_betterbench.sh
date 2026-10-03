#!/usr/bin/env bash
# Run sequentially: one resident NInfer model, separate reports for its two presets.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
thor_dir="${NINFER_THOR_DIR:-$HOME/ninfer-thor}"
report_dir="${1:-$thor_dir/reports/betterbench-$(date -u +%Y%m%d-%H%M%S)}"
python="$thor_dir/betterbench-venv/bin/python"
mkdir -p "$report_dir"
cp "$script_dir/betterbench_thor.json" "$report_dir/config.json"
git -C "$thor_dir/BetterBench" rev-parse HEAD > "$report_dir/betterbench-revision.txt"
docker image inspect ninfer-thor:sm110 --format '{{.Id}}' > "$report_dir/ninfer-image.txt"
ready() {
  local limit=$((SECONDS + 120))
  until curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do
    if (( SECONDS >= limit )); then return 1; fi
    sleep 1
  done
}
common=(run --endpoint http://127.0.0.1:8000/v1 --model unsloth/Qwen3.8-27B-NVFP4
  --config "$script_dir/betterbench_thor.json" --no-update-check
  --note "revision=${NINFER_REVISION:-unknown}" --note kv=fp8 --note thinking=false
  --note clocks=jetson_clocks
  --note "artifact=${NINFER_ARTIFACT:-/work/models/qwen3_8_27b_thor_dflash2.ninfer}")
bash "$thor_dir/run_ninfer_thor.sh" single
ready
"$python" -u "$script_dir/betterbench_ninfer.py" "${common[@]}" \
  --decode --note preset=single --note drafts=15 --out "$report_dir/single.json" \
  2>&1 | tee "$report_dir/single.log"
# Include the tokenizer-calibrated 50K agent-history prompt from the tuning run.
# Prefix reuse models an agent continuing its history; prefill below stays cold.
if [[ -f "$thor_dir/deep-context-50k-prompt.json" ]]; then
  mkdir -p "$report_dir/deep-corpus"
  "$python" - "$thor_dir/deep-context-50k-prompt.json" "$report_dir" <<'PYTHON'
import json
import pathlib
import sys
prompt = json.loads(pathlib.Path(sys.argv[1]).read_text())
report = pathlib.Path(sys.argv[2])
row = {"id": "agent-history-50k", "category": "agentic50k",
       "input_len_bucket": "long", "output_len_bucket": "medium",
       "max_tokens": 256, "messages": [{"role": "user", "content": prompt["prompt"]}]}
(report / "deep-corpus/agentic50k.jsonl").write_text(json.dumps(row) + "\n")
config = json.loads((report / "config.json").read_text())
config["unique_nonce"] = False
config["weights"] = {"agentic50k": 1.0}
(report / "deep-config.json").write_text(json.dumps(config, indent=2) + "\n")
PYTHON
  "$python" -u "$script_dir/betterbench_ninfer.py" "${common[@]}" \
    --config "$report_dir/deep-config.json" --decode \
    --corpus "$report_dir/deep-corpus" --categories agentic50k \
    --note preset=single --note drafts=15 --note prefix=cached \
    --out "$report_dir/agentic50k.json" 2>&1 | tee "$report_dir/agentic50k.log"
  NINFER_DRAFT_TOKENS=5 bash "$thor_dir/run_ninfer_thor.sh" single
  ready
  "$python" -u "$script_dir/betterbench_ninfer.py" "${common[@]}" \
    --config "$report_dir/deep-config.json" --decode \
    --corpus "$report_dir/deep-corpus" --categories agentic50k \
    --note preset=single --note drafts=5 --note prefix=cached \
    --out "$report_dir/agentic50k-k5.json" 2>&1 | tee "$report_dir/agentic50k-k5.log"
fi
bash "$thor_dir/run_ninfer_thor.sh" multi
ready
"$python" -u "$script_dir/betterbench_ninfer.py" "${common[@]}" \
  --decode --prefill --concurrency --note preset=multi --note drafts=9 \
  --out "$report_dir/multi.json" 2>&1 | tee "$report_dir/multi.log"
printf 'Reports: %s/single.html and %s/multi.html\n' "$report_dir" "$report_dir"
