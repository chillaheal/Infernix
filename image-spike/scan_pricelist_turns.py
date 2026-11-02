"""Scan one Claude Code session transcript for the reported symptoms.

Symptom 1: an assistant turn that announces a tool run in text and then ends with a terminal stop
reason and NO tool_use block. Symptom 2: the same user text delivered repeatedly. Symptom 3: the
model label in the environment block versus the model id the transcript records.

Usage: py -3.13 -I scan_pricelist_turns.py <transcript.jsonl>
"""
import json
import re
import sys

ANNOUNCES = re.compile(
    r"(nu kör jag|nu kor jag|kör jag|ska köra|ska kor|nu startar|starts now|i och med att jag kör"
    r"|now i (?:will|run)|let me run|i will run)", re.I)
TERMINAL = {"end_turn", "stop_sequence", "stop", "max_tokens", "length"}


def blocks(msg):
    content = msg.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [b for b in content if isinstance(b, dict)]
    return []


def main(path):
    stops = {}
    suspicious = []
    repeats = {}
    labels = {}
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = str(rec.get("timestamp", ""))[:19]
            kind = rec.get("type")
            msg = rec.get("message") or {}
            if kind == "assistant" and isinstance(msg, dict):
                stop = str(msg.get("stop_reason"))
                stops[stop] = stops.get(stop, 0) + 1
                parts = blocks(msg)
                text = "".join(b.get("text", "") for b in parts if b.get("type") == "text")
                thinking = "".join(b.get("thinking", "") for b in parts if b.get("type") == "thinking")
                has_tool = any(b.get("type") == "tool_use" for b in parts)
                tail = (text or thinking).strip()[-110:]
                if not has_tool and stop in TERMINAL and ANNOUNCES.search(text or ""):
                    suspicious.append((lineno, ts, stop, repr(tail)))
            elif kind == "user":
                parts = blocks(msg) if isinstance(msg, dict) else []
                text = " ".join(b.get("text", "") for b in parts if b.get("type") == "text")
                if "pausade" in text.lower():
                    repeats.setdefault(text.strip()[:60], []).append(ts)
                for m in re.finditer(r"powered by the model named ([A-Za-z0-9._-]+)", text):
                    labels[m.group(1)] = labels.get(m.group(1), 0) + 1
                for m in re.finditer(r"exact model (?:ID|id)[:\s]*([A-Za-z0-9._-]+)", text):
                    key = m.group(1)
                    labels[key] = labels.get(key, 0) + 1
            model = rec.get("model") or (msg.get("model") if isinstance(msg, dict) else None)
            if isinstance(model, str) and model:
                labels["record:" + model] = labels.get("record:" + model, 0) + 1

    print("stop_reason counts:", json.dumps(stops, sort_keys=True))
    print()
    print("model labels seen (name or record:<id> -> count):")
    for key, count in sorted(labels.items(), key=lambda kv: -kv[1]):
        print("  %-46s %d" % (key, count))
    print()
    print("repeated user text containing 'pausade':")
    for key, times in repeats.items():
        print("  x%d  %s" % (len(times), key))
        for t in times:
            print("        %s" % t)
    print()
    print("turns that announce a tool run yet end terminal with no tool_use: %d" % len(suspicious))
    for lineno, ts, stop, tail in suspicious[:40]:
        print("  line %-7d %s stop=%-13s tail=%s" % (lineno, ts, stop, tail))


if __name__ == "__main__":
    main(sys.argv[1])
