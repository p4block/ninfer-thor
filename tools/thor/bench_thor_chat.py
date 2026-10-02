"""Warmed streaming chat benchmark; standard-library client runs on Thor."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import statistics
import time
import urllib.request

PROMPTS = {
    "code": "Write a Python implementation of merge sort with type hints, explain its correctness and complexity, and include several tests.",
    "chat": "Explain how matrix multiplication uses a GPU in plain English, then work through a numerical example and explain memory bandwidth and batching.",
}


def run(base, model, prompt, tokens, seed):
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "top_p": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "seed": seed, "max_tokens": tokens, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    usage = None
    parts = []
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    parts.append(piece)
    end = time.perf_counter()
    if not usage or first is None or last <= first:
        raise RuntimeError({"usage": usage, "text": "".join(parts)})
    count = usage["completion_tokens"]
    return {"tokens": count, "prompt_tokens": usage.get("prompt_tokens"), "ttft_s": first-start,
            "elapsed_s": end-start, "decode_tps": (count-1)/(last-first), "text": "".join(parts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-NVFP4")
    parser.add_argument("--label", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    for name, prompt in PROMPTS.items():
        run(args.base, args.model, prompt, 32, 1234)
        batches = []
        for repetition in range(args.reps):
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                requests = list(pool.map(lambda lane: run(args.base, args.model, prompt, args.tokens,
                                                          1234 + repetition + lane),
                                         range(args.concurrency)))
            batches.append({"requests": requests, "aggregate_tps":
                            sum(row["tokens"] for row in requests)/(time.perf_counter()-start)})
        rows = [row for batch in batches for row in batch["requests"]]
        print(json.dumps({"label": args.label, "prompt": name, "concurrency": args.concurrency,
                          "tokens_requested": args.tokens, "reps": args.reps,
                          "decode_tps_median": statistics.median(row["decode_tps"] for row in rows),
                          "ttft_s_median": statistics.median(row["ttft_s"] for row in rows),
                          "aggregate_tps_median": statistics.median(batch["aggregate_tps"] for batch in batches),
                          "batches": batches}), flush=True)


if __name__ == "__main__":
    main()
