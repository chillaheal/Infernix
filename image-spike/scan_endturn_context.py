"""Group a transcript by API turn and report every turn that ended without emitting a tool call.

Streaming stores one assistant record per content block, so the turn is the requestId, not the record.
Each turn is judged on its whole: the blocks it carried, whether any was tool_use, the joined text, and
which records came right after it (redelivered user message, queue remove, retry error, compaction).

Usage: py -3.13 -I scan_endturn_context.py <transcript.jsonl> [model_filter]
"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def text_of(msg):
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text")
    return ""


def kinds_of(msg):
    content = msg.get("content")
    if isinstance(content, list):
        return [str(b.get("type")) for b in content if isinstance(b, dict)]
    return ["str"] if isinstance(content, str) else []


def describe(rec):
    kind = str(rec.get("type"))
    msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
    if kind in ("user", "assistant") and msg:
        body = text_of(msg)
        return "%s %s stop=%s %r" % (kind, msg.get("role"), msg.get("stop_reason"), body[:120])
    if kind == "queue-operation":
        return "queue %s/%s %r" % (rec.get("operation"), rec.get("reason"), str(rec.get("content"))[:80])
    if kind == "system":
        return "system level=%s retry=%s %r" % (rec.get("level"), rec.get("retryAttempt"),
                                                str(rec.get("content") or rec.get("error"))[:90])
    return kind


def main(path, want=""):
    records = []
    for line in open(path, encoding="utf-8"):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            records.append(None)

    turns = {}                      # requestId -> (first_index, blocks, text, stop, model)
    order = []
    for i, rec in enumerate(records):
        if not rec or rec.get("type") != "assistant":
            continue
        msg = rec.get("message") or {}
        key = str(rec.get("requestRef") or rec.get("requestId") or msg.get("id") or i)
        if key not in turns:
            turns[key] = {"i": i, "kinds": [], "text": "", "stop": None,
                          "model": str(rec.get("requestedModel")), "ts": str(rec.get("timestamp", ""))}
            order.append(key)
        slot = turns[key]
        slot["kinds"] += kinds_of(msg)
        slot["text"] += text_of(msg)
        if msg.get("stop_reason"):
            slot["stop"] = msg.get("stop_reason")

    shown = 0
    for key in order:
        slot = turns[key]
        if want and slot["model"] != want:
            continue
        if "tool_use" in slot["kinds"]:
            continue
        shown += 1
        i = slot["i"]
        print("=== %s model=%s stop=%s blocks=%s" % (slot["ts"][:19], slot["model"], slot["stop"],
                                                     ",".join(sorted(set(slot["kinds"])))))
        print("    text tail: %r" % slot["text"].strip()[-240:])
        printed = 0
        for j in range(i + 1, len(records)):
            rec = records[j]
            if not rec:
                continue
            if rec.get("type") == "assistant" and str(rec.get("requestRef") or
                                                       rec.get("requestId")) == key:
                continue
            print("    next: %s" % describe(rec))
            printed += 1
            if printed >= 3:
                break
        print()
    print("turns with no tool_use under model filter %r: %d" % (want or "*", shown))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")

