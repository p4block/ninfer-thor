#!/usr/bin/env python3
"""Adapt requests to NInfer; keep BetterBench's timers and metrics unchanged."""
import json
import sys
import urllib.request
from betterbench import client, runner
from betterbench.cli import main

original_payload = client._build_payload
original_prefill_messages = runner.make_prefill_messages


def ninfer_payload(*args, **kwargs):
    payload = original_payload(*args, **kwargs)
    payload['chat_template_kwargs'] = {'enable_thinking': False}
    return payload


def calibrate_prefill(endpoint, model):
    def messages(target_tokens, nonce):
        # BetterBench's four-chars/token estimate undershoots on this tokenizer.
        # Count before the timed request; retain its fresh randomized body/nonce.
        estimate = target_tokens
        for _ in range(3):
            result = original_prefill_messages(estimate, nonce)
            payload = {'model': model, 'messages': result,
                       'thinking': {'type': 'disabled'}}
            request = urllib.request.Request(
                endpoint.rstrip('/') + '/messages/count_tokens',
                json.dumps(payload).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=120) as response:
                actual = json.load(response)['input_tokens']
            if abs(actual - target_tokens) <= max(32, target_tokens // 1000):
                return result
            estimate = max(64, round(estimate * target_tokens / actual))
        return result
    runner.make_prefill_messages = messages


client._build_payload = ninfer_payload
if __name__ == '__main__':
    if '--endpoint' in sys.argv and '--model' in sys.argv:
        calibrate_prefill(sys.argv[sys.argv.index('--endpoint') + 1],
                          sys.argv[sys.argv.index('--model') + 1])
    main()
