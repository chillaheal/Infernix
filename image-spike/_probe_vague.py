"""Does the model ask a follow-up when the picture request is vague, given the generate_image tool?

Deliberately under-specified prompts, empty session, the same tool definition as _probe_self_generate.py.
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8888"
PROMPTS = ["gor en bild", "mala nagot fint"]

TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Create a picture from a text description and return it.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "What to draw"},
                "size": {"type": "string", "description": "WIDTHxHEIGHT, e.g. 1024x1024"},
            },
            "required": ["prompt"],
        },
    },
}

model = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=30))["data"][0]["id"]

for prompt in PROMPTS:
    body = {
        "model": model,
        "max_tokens": 700,
        "temperature": 0,
        "tools": [TOOL],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": prompt}],
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        message = json.loads(response.read())["choices"][0]["message"]
    called = message.get("tool_calls")
    lines = [
        "PROMPT: " + prompt,
        "  tool_call: " + (json.dumps(called, ensure_ascii=False)[:400] if called else "NONE"),
        "  text: " + json.dumps(message.get("content"), ensure_ascii=False)[:500],
        "  reasoning: " + json.dumps(message.get("reasoning_content"), ensure_ascii=False)[-400:],
    ]
    # The Windows console is cp1252; write the Swedish answers to a UTF-8 file instead of printing.
    with open("_probe_vague.out.txt", "a", encoding="utf-8") as sink:
        sink.write("\n".join(lines) + "\n")
    print("PROMPT:", prompt, "tool_call:", "YES" if called else "NONE")
