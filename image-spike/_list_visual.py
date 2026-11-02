"""Print the distinct visual tensor name patterns of a Qwen3-VL text encoder index."""
import json
import re
import sys

index = sys.argv[1]
data = json.load(open(index, encoding="utf-8"))
keys = [k for k in data["weight_map"] if "visual" in k]
print("n_visual", len(keys))
patterns = sorted({re.sub(r"\.\d+\.", ".<i>.", k) for k in keys})
for pattern in patterns:
    print(pattern)
