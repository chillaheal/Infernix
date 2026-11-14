"""Read the harness's OWN records out of one Claude Code transcript.

The transcript stores more than the conversation: system records carry retry bookkeeping
(retryAttempt / maxRetries / preventedContinuation / error), queue-operation records carry the
delivery of a user message (operation / reason / deliveryId), and assistant records carry
requestedModel. That is the harness's own account of a clipped turn and a redelivered message, which
is what the symptoms describe.

Usage: py -3.13 -I scan_harness_events.py <transcript.jsonl>
"""
import collections
import json
import sys


def main(path):
    retries = []
    queue_ops = collections.Counter()
    queue_first = {}
    models = collections.defaultdict(lambda: [None, None, 0])
    stop_by_model = collections.defaultdict(collections.Counter)
    api_params = collections.Counter()

    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = str(rec.get("type"))
            ts = str(rec.get("timestamp", ""))[:19]

            if kind == "system":
                touched = [k for k in ("retryAttempt", "maxRetries", "preventedContinuation",
                                       "error", "level") if rec.get(k) not in (None, "", False, 0)]
                if touched:
                    retries.append((lineno, ts, ",".join(touched), str(rec.get("content"))[:150]))

            elif kind == "queue-operation":
                op = "%s|%s" % (rec.get("operation"), rec.get("reason"))
                queue_ops[op] += 1
                queue_first.setdefault(op, []).append((ts, str(rec.get("content"))[:60]))

            elif kind == "assistant":
                model = str(rec.get("requestedModel") or "?")
                slot = models[model]
                if slot[0] is None or ts < slot[0]:
                    slot[0] = ts
                if slot[1] is None or ts > slot[1]:
                    slot[1] = ts
                slot[2] += 1
                stop = str((rec.get("message") or {}).get("stop_reason"))
                stop_by_model[model][stop] += 1

            elif kind == "api-request":
                params = rec.get("params") or {}
                api_params["%s|%s|%s" % (rec.get("base"), params.get("model"),
                                         str(params.get("max_tokens")))] += 1

    print("retry / continuation records (system): %d" % len(retries))
    for lineno, ts, touched, content in retries[:40]:
        print("  line %-8d %s %-40s %s" % (lineno, ts, touched, content.replace("\n", " ")))
    if len(retries) > 40:
        print("  ... %d more" % (len(retries) - 40))
    print()
    print("queue-operation by operation|reason:")
    for key, count in queue_ops.most_common():
        firsts = queue_first[key][:3]
        print("  x%-4d %-34s e.g. %s" % (count, key, "; ".join("%s %r" % f for f in firsts)))
    print()
    print("requestedModel -> first/last turn, count, stop reasons:")
    for model, (first, last, count) in sorted(models.items()):
        print("  %-22s %s .. %s  n=%d  %s" % (model, first, last, count,
                                              dict(stop_by_model[model])))
    print()
    print("api-request base|model|max_tokens:")
    for key, count in api_params.most_common(12):
        print("  x%-4d %s" % (count, key))


if __name__ == "__main__":
    main(sys.argv[1])
