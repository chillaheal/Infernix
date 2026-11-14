"""Where does a phrase actually occur inside a transcript?

For every JSONL record whose raw line contains the needle, print what KIND of record it is, so a
genuine repeated user delivery can be told apart from the same words echoed inside a compaction
summary, a tool result, or a pasted report.

Usage: py -3.13 -I scan_repeat_delivery.py <transcript.jsonl> <needle>
"""
import json
import sys


def kinds(msg, needle):
    """Return the block kinds in a message where the needle actually appears."""
    found = set()
    content = msg.get("content")
    if isinstance(content, str):
        if needle in content:
            found.add("text")
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            blob = json.dumps(block, ensure_ascii=False)
            if needle in blob:
                kind = str(block.get("type"))
                if kind == "tool_result":
                    found.add("tool_result")
                elif kind == "text":
                    found.add("text")
                else:
                    found.add(kind + "(other)")
    return found


def main(path, needle):
    exact_user_turns = []
    other = {}
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if needle not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
            where = kinds(msg, needle) if msg else set()
            flags = [k for k in ("isCompactSummary", "isSidechain", "isMeta", "isVisibleInTranscriptOnly")
                     if rec.get(k)]
            role = str(msg.get("role"))
            text = ""
            if where == {"text"}:
                text = "".join(b.get("text", "") for b in (msg.get("content") or [])
                               if isinstance(b, dict) and b.get("type") == "text").strip()
            if role == "user" and where == {"text"} and len(text) < 120:
                exact_user_turns.append((lineno, str(rec.get("timestamp", ""))[:19], text, flags))
            else:
                key = (str(rec.get("type")), role, ",".join(sorted(where)) or "outside-message",
                       ",".join(flags))
                other[key] = other.get(key, 0) + 1

    print("SHORT USER TURNS that are the needle itself: %d" % len(exact_user_turns))
    for lineno, ts, text, flags in exact_user_turns:
        print("  line %-8d %s flags=%-22s text=%r" % (lineno, ts, ",".join(flags) or "-", text[:80]))
    print()
    print("every other place the needle appears (type, role, block kind, flags -> count):")
    for (typ, role, where, flags), count in sorted(other.items(), key=lambda kv: -kv[1]):
        print("  x%-4d type=%-9s role=%-7s where=%-22s flags=%s" % (count, typ, role, where, flags or "-"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
