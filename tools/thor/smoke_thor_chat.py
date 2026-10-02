"""Small generation, Unicode, tool-use, and context retrieval smoke checks."""
import json
import time
import urllib.request


def chat(name, messages, max_tokens=64, **kwargs):
    payload = dict(model="unsloth/Qwen3.8-27B-NVFP4", messages=messages,
                   max_tokens=max_tokens, temperature=0, presence_penalty=0,
                   chat_template_kwargs={"enable_thinking": False}, **kwargs)
    start = time.perf_counter()
    request = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                                     json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    print(json.dumps({"case": name, "elapsed_s": time.perf_counter()-start,
                      "response": result}), flush=True)
    return result["choices"][0]["message"]


math = chat("arithmetic", [{"role": "user", "content":
    "What is 17 * 19? Respond with the integer only."}])
assert math["content"].strip() == "323", math
unicode = chat("unicode", [{"role": "user", "content":
    "请只回答：北京是哪个国家的首都？"}])
assert "中国" in unicode["content"], unicode
tool = chat("tool_call", [{"role": "user", "content":
    "Use the get_temperature tool to get the temperature in Madrid."}], tools=[{
        "type": "function", "function": {"name": "get_temperature",
        "description": "Get current temperature for a city", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}},
            "required": ["city"]}}}], max_tokens=128)
assert tool.get("tool_calls"), tool
call = tool["tool_calls"][0]["function"]
assert call["name"] == "get_temperature", call
assert "Madrid" in json.loads(call["arguments"])["city"], call
filler = "The unrelated inventory has blue pens, green folders, and wooden tables.\n" * 600
context = chat("context_retrieval", [{"role": "user", "content":
    filler[:len(filler)//2] + "\nThe secret access code is THOR-7391-ZEBRA.\n" +
    filler[len(filler)//2:] + "\nWhat is the secret access code? Answer only that code."}])
assert "THOR-7391-ZEBRA" in context["content"], context
print(json.dumps({"status": "all four smoke checks passed"}), flush=True)
