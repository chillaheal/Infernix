import sys, json, base64, urllib.request, urllib.error

def main():
    base = "http://127.0.0.1:8888"
    prompt = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else "768x768"
    out = sys.argv[3] if len(sys.argv) > 3 else "image.png"
    with urllib.request.urlopen(base + "/v1/models", timeout=15) as r:
        mid = json.load(r)["data"][0]["id"]
    body = json.dumps({"model": mid, "prompt": prompt, "size": size, "n": 1,
                       "response_format": "b64_json"}).encode()
    req = urllib.request.Request(base + "/v1/images/generations", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.load(r)
        b = base64.b64decode(data["data"][0]["b64_json"])
        with open(out, "wb") as f:
            f.write(b)
        print("MODEL=" + mid + " HTTP=200 SIZE=" + size + " BYTES=" + str(len(b)) + " FILE=" + out)
    except urllib.error.HTTPError as e:
        print("HTTP=" + str(e.code) + " BODY=" + e.read().decode("utf-8", "replace")[:400])

main()
