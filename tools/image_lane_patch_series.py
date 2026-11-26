#!/usr/bin/env python3
"""Cut the image lane out of the fork's diff against upstream: one lane-only patch per upstream-owned file.

"git diff <upstream> HEAD -- <path>" on this fork is NOT the lane. The fork also carries unrelated
chat-model work and trails upstream's own refactors, so the whole-file diff is ~5900 added lines of
which only ~11% mention the lane at all; applying it overwrites upstream's newer code with the fork's
older shape (the first lane-only port died compiling include/infernix/types.h, because the fork's
KvCacheStorage/FastPath/ContextCacheOptions took the place of upstream's kv_format.h refactor).

This tool takes the diff and keeps only the hunks that belong to the lane: a hunk is kept when one of
its added or removed lines matches a lane token. Whole hunks are kept or dropped, so the surviving
offsets stay exactly as git produced them and "git apply --3way" reads them normally. Every decision is
printed with its hunk header under --verbose so a person can audit what crossed and what did not.

Usage:
  py -3.13 -I tools/image_lane_patch_series.py --upstream <sha> [--our-ref HEAD] --plan plan.json \
      --out <dir> [--verbose] [--tokens extra,regex]

--plan is the JSON tools/check_image_seams.py --emit-plan wrote; rows with category ours-only are skipped
(those files are copied, not patched). Exit 0 always unless a git call fails: a row that yields no lane
hunk is a normal outcome and is reported as such.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

# What makes a changed line belong to the image lane. Deliberately literal: these are the names the
# lane's own files and seams use. Anything not in here is fork drift or upstream code and must not
# cross into a lane-only port. Extend with --tokens only after reading the hunk by hand.
LANE_TOKENS = [
    r"image", r"Image", r"IMAGE", r"chat_image", r"prompt_enhancer", r"PromptEnhancer", r"enhancer",
    r"\bvae", r"Vae", r"VAE", r"\bdit", r"Dit", r"DIT", r"qwen_image", r"artifact_constants",
    r"reference_image", r"openai_images", r"png_encode", r"rope_c64", r"attn_dit", r"adaln", r"Adaln",
    r"text_tower", r"vision_tower", r"joint_layout", r"embed_gather", r"img_mask", r"image_job",
    r"Images", r"artifact_url", r"media_encode", r"[Ii]nterlude", r"device_need", r"DeviceNeed",
    r"vram_ledger", r"VRAM", r"infernix_qwen_image", r"infernix_media_encode", r"ChatImage",
]

# Deny list, per upstream-owned file: a hunk that matches a lane token AND one of these belongs to the
# fork's unrelated chat-model work, not to the lane, and is dropped. The hunks come from the fork's diff
# against upstream tip, and this fork trails upstream's own refactors on these exact files -- a kept
# hunk there carries the fork's shape of upstream code along with the lane line, which is how the first
# two lane-only ports died (types.h's KvCacheStorage/ThinkingBudgetPolicy over upstream's kv_format.h
# refactor, engine_core.h's maintenance waker over upstream's EngineCore).
#
# This table is reviewed residue, not a guess: each entry was read in the hunk that carries it
# (--verbose prints every hunk with its header and verdict). Add to it when a kept hunk drags another
# fork-only declaration across; never widen it to a whole file, or the lane's own hunk disappears too.
FORK_ONLY = {
    "include/infernix/types.h": [
        r"KvCacheStorage", r"ThinkingBudgetPolicy", r"thinking_budget_policy",
        r"kDefaultRamHeadroomBytes", r"ram_headroom", r"HostContextBytes", r"host_context",
        r"expert_stripe", r"ExpertStripe",
    ],
    "src/runtime/engine/engine_core.h": [
        r"kProgramMaintains", r"set_maintenance_waker", r"submit_door_open", r"FinishReason::Cancelled",
        r"abort\(", r"Degraded",
    ],
    "src/serve/serve_options.h": [
        r"ram_headroom", r"ram-headroom", r"thinking-budget", r"thinking_budget",
        r"host-context", r"host_context", r"expert-stripe", r"kv-dtype",
    ],
    "src/serve/serve_options.cpp": [
        r"ram_headroom", r"ram-headroom", r"thinking-budget", r"thinking_budget",
        r"host-context", r"host_context", r"expert-stripe", r"kv-dtype",
    ],
    "src/models/qwen4_exp/program/program.cpp": [r"expert_stripe", r"ThinkingBudget", r"Vq2", r"Q4Key"],
    "src/models/qwen4_exp/program/program.h": [r"expert_stripe", r"ThinkingBudget", r"Vq2", r"Q4Key"],
    "src/models/qwen4_exp/program/program_impl.h": [r"expert_stripe", r"ThinkingBudget", r"Vq2", r"Q4Key"],
    "src/runtime/expert_offload/expert_residency.cpp": [r"expert_stripe", r"Vq2", r"Q4Key", r"stripe"],
    "tests/models/qwen4_exp/test_memory_plan.cpp": [r"expert_stripe", r"ThinkingBudget", r"Vq2", r"Q4Key"],
    "tests/ops/linear/test_fp8_a16.cpp": [r"q56", r"Q56", r"noalias"],
}

HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
LANE_PATTERN = None            # set in main(): the compiled lane token alternation


def git_diff(repo, upstream, our_ref, path):
    argv = ["git", "-C", repo, "diff", "--no-ext-diff", upstream]
    if our_ref:
        argv.append(our_ref)
    argv += ["--", path]
    out = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8")
    if out.returncode not in (0, 1):          # 1 = differences found
        raise SystemExit("git diff failed for %s: %s" % (path, out.stderr.strip()))
    return out.stdout


def split_patch(text):
    """Return (header_lines, [hunk_lines, ...]) for a single-file unified diff."""
    lines = text.splitlines(keepends=True)
    header, hunks, cur = [], [], None
    for ln in lines:
        if HUNK_RE.match(ln):
            if cur is not None:
                hunks.append(cur)
            cur = [ln]
        elif cur is None:
            header.append(ln)
        else:
            cur.append(ln)
    if cur is not None:
        hunks.append(cur)
    return header, hunks


def changed_lines(hunk):
    return [ln[1:] for ln in hunk[1:] if ln.startswith("+") or ln.startswith("-")]


def hunk_is_lane(hunk, pattern):
    return any(pattern.search(ln) for ln in changed_lines(hunk))


def hunk_is_fork_only(hunk, deny):
    """Reject a hunk when it deletes upstream code that has nothing to do with the lane, or when all of
    its lane-looking lines are marked fork-only for that file.

    The rule that matters: this fork trails upstream's own refactors of the shared contracts, so a
    hunk taken from the fork diff routinely DELETES upstream lines the fork never saw. Deleting
    unrelated upstream code is never the lane's business, so any hunk whose deleted lines are not
    lane-related is dropped -- that is what kept the fork's KvCacheStorage/ThinkingBudgetPolicy shape
    from replacing upstream's kv_format.h refactor, and the fork's maintenance waker from replacing
    upstream's EngineCore. A hunk that only adds, or only removes lane-related lines, is the lane."""
    removed = [ln[1:] for ln in hunk[1:] if ln.startswith("-") and not ln.startswith("---")]
    if any(not LANE_PATTERN.search(ln) for ln in removed):
        return True
    added = [ln[1:] for ln in hunk[1:] if ln.startswith("+") and not ln.startswith("+++")]
    return bool(deny) and any(any(d.search(ln) for d in deny) for ln in added)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--our-ref", default="HEAD",
                    help="ref to diff against (default: %(default)s); ignored with --working-tree")
    ap.add_argument("--working-tree", action="store_true",
                    help="diff the working tree instead of a ref -- this is what carries uncommitted "
                         "seam work across (an empty --our-ref cannot survive a shell, so this is a flag)")
    ap.add_argument("--plan", required=True, help="JSON written by tools/check_image_seams.py --emit-plan")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", default="", help="extra comma-separated regexes treated as lane code")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    global LANE_PATTERN
    tokens = list(LANE_TOKENS)
    if args.tokens:
        tokens += [t for t in args.tokens.split(",") if t.strip()]
    pattern = re.compile("|".join(tokens))
    LANE_PATTERN = pattern

    with open(args.plan, encoding="utf-8") as fh:
        rows = json.load(fh).get("rows", [])
    rows = [r for r in rows if r.get("category") != "ours-only" and r.get("upstream_path")]
    os.makedirs(args.out, exist_ok=True)

    kept_rows, empty_rows, dropped_hunks, kept_hunks = [], [], 0, 0
    for index, row in enumerate(sorted(rows, key=lambda r: r["upstream_path"]), start=1):
        path = row["upstream_path"]
        our_ref = None if args.working_tree else args.our_ref
        text = git_diff(repo, args.upstream, our_ref, path)
        if not text.strip():
            empty_rows.append(path)
            continue
        header, hunks = split_patch(text)
        deny = [re.compile(d) for d in FORK_ONLY.get(path, [])]
        kept = [h for h in hunks if hunk_is_lane(h, pattern) and not hunk_is_fork_only(h, deny)]
        dropped_hunks += len(hunks) - len(kept)
        kept_hunks += len(kept)
        if args.verbose:
            for h in hunks:
                print("%-56s %s %s" % (path, h[0].split("@@")[1].strip(),
                                       "KEEP" if h in kept else "drop"))
        if not kept:
            empty_rows.append(path)
            continue
        name = "%03d-%s.patch" % (index, path.replace("/", "_"))
        with open(os.path.join(args.out, name), "w", encoding="utf-8", newline="") as fh:
            fh.write("".join(header))
            for h in kept:
                fh.write("".join(h))
        kept_rows.append((path, len(kept), len(hunks) - len(kept)))

    print()
    print("hunks kept %d, dropped %d across %d row(s); %d row(s) carry no lane change"
          % (kept_hunks, dropped_hunks, len(kept_rows), len(empty_rows)))
    for path, kept, dropped in kept_rows:
        print("  %-56s %2d kept / %2d dropped" % (path, kept, dropped))
    if empty_rows:
        print("  no lane hunk: %s" % ", ".join(empty_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
