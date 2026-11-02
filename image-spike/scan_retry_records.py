"""Print the harness's retry/continuation records verbatim, inside an optional time window.

Each `system` record that carries retryAttempt / maxRetries / preventedContinuation / error is printed
with every field that has a value, so the API error the harness saw is visible rather than inferred.

Usage: py -3.13 -I scan_retry_records.py <transcript.jsonl> [since] [until]
"""
import json
import sys


def main(path, since="", until=""):
    shown = 0
    total = 0
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "system":
                continue
            fields = {k: v for k, v in rec.items()
                      if k in ("retryAttempt", "maxRetries", "preventedContinuation", "error",
                               "level", "content", "hasOutput", "hookErrors", "timestamp",
                               "logicalParentUuid")}
            if not any(rec.get(k) not in (None, "", False, 0)
                       for k in ("retryAttempt", "maxRetries", "preventedContinuation", "error")):
                continue
            total += 1
            ts = str(rec.get("timestamp", ""))
            if since and ts < since:
                continue
            if until and ts > until:
                continue
            shown += 1
            print("line %d %s" % (lineno, ts))
            for key in ("retryAttempt", "maxRetries", "preventedContinuation", "level", "error",
                        "content", "hasOutput"):
                value = fields.get(key)
                if value in (None, "", False, 0):
                    continue
                text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                print("    %-22s %s" % (key, text[:600].replace("\n", " | ")))
            print()
    print("retry-bearing system records total=%d shown_in_window=%d" % (total, shown))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "", sys.argv[3] if len(sys.argv) > 3 else "")
