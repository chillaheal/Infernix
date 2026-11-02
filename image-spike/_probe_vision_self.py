"""Does the served model know it has VISION, and does the input path work, in an empty session?

V1: text only, ask whether it can see images.
V2: an actual PNG attached as an OpenAI image_url part; see whether it describes the picture.
"""
import base64
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8888"
PNG = sys.argv[1]

model = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=30))["data"][0]["id"]


def chat(messages):
    body = {"model": model, "max_tokens": 700, "temperature": 0, "messages": messages}
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.loads(response.read())


out = []
answer = chat([{"role": "user", "content": "kan du se bilder som jag skickar i chatten?"}])
message = answer["choices"][0]["message"]
out.append("V1_CLAIM text: " + json.dumps(message.get("content"), ensure_ascii=False))
out.append("V1_CLAIM reasoning: " + json.dumps(message.get("reasoning_content"), ensure_ascii=False)[-400:])

blob = base64.b64encode(open(PNG, "rb").read()).decode()
answer = chat([{
    "role": "user",
    "content": [
        {"type": "text", "text": "vad visar denna bild? svara kort."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + blob}},
    ],
}])
message = answer["choices"][0]["message"]
out.append("V2_IMAGE_USED text: " + json.dumps(message.get("content"), ensure_ascii=False))
out.append("V2_USAGE: " + json.dumps(answer.get("usage"), ensure_ascii=False))

with open("_probe_vision_self.out.txt", "w", encoding="utf-8") as sink:
    sink.write("\n\n".join(out) + "\n")
print("wrote _probe_vision_self.out.txt")
