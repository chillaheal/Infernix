"""Ask the running server, in an EMPTY session, whether it generates an image on its own.

Case A: plain chat, no tools. Case B: the same turn with a generate_image tool definition.
Nothing here changes the server; it only reads what the model answers.
"""
import json
import urllib.request

BASE = "http://127.0.0.1:8888"
PROMPT = "skapa bild av ett bord"


def post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.loads(response.read())


model = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=30))["data"][0]["id"]
print("MODEL", model)

common = dict(model=model, max_tokens=200, temperature=0, messages=[{"role": "user", "content": PROMPT}])

answer = post("/v1/chat/completions", common)["choices"][0]["message"]
print("A_NO_TOOL finish/content:", json.dumps(answer, ensure_ascii=False)[:600])

tool = {
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
with_tool = post("/v1/chat/completions", dict(common, tools=[tool], tool_choice="auto"))["choices"][0]["message"]
print("B_WITH_TOOL:", json.dumps(with_tool, ensure_ascii=False)[:900])
