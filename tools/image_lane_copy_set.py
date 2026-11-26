#!/usr/bin/env python3
"""Derive the image lane's ours-only copy set from CMake's own generated build files.

The port driver (tools/port_lane.ps1) copies files. Which files? Not a hand-kept list: this tool asks
the build system. It reads the lane's own ours-only .cmake files, takes the TARGET NAMES out of them,
reads each target's generated <target>.vcxproj under the build directory (CMake's own output: the exact
ClCompile/ClInclude/CudaCompile/None item set and the AdditionalIncludeDirectories), and closes the
quoted/angle #include graph over those include directories. Everything the result contains that the
upstream ref lacks is a file the port must copy -- headers, CUDA kernels, fixtures and Python tools
included. A target whose sources are all ours-only is entirely lane and is taken whole; a target
upstream also builds is filtered to the files a lane cmake file actually names, so the port carries
the lane and not the fork's unrelated work.

Usage:
  py -3.13 -I tools/image_lane_copy_set.py --upstream <sha> --build-dir build-windows \
      [--write-json docs/maintainer/image-lane.copy-set.json] [--check]

--check compares the committed JSON with a fresh derivation and exits 1 on drift (a lane file was added,
deleted or newly reached) or when a listed file is gone. Exit 0 means the copy set is current.
"""

import argparse
import json
import os
import posixpath
import re
import subprocess
import sys

# Ours-only files the derivation reaches but the port must NOT copy, with the reason. Each was checked
# by hand: no lane source includes them, and upstream tip already owns the thing they stand in for.
EXCLUSIONS = {
    "src/models/qwen3_5/frontend/tokenizer.h":
        "fork-side path; upstream tip owns the tokenizer at src/text/tokenizer.h",
    "src/runtime/expert_offload/expert_profile.h":
        "fork-side expert profiling, not reached by any lane source",
    "tests/convert/test_expert_profile.py":
        "python test for the fork-side expert profiler, no lane target uses it",
}

SOURCE_EXT = (".cpp", ".cu", ".c", ".cc", ".h", ".hpp", ".cuh")
COMPILE_EXT = (".cpp", ".cu", ".c", ".cc")
RUNTIME_EXT = (".json", ".py")
# Runtime dependencies a lane process invokes rather than includes: the prompt-enhancer sidecar the
# serve launches, and the converter the artifact recipe documents.
EXPLICIT = ("tools/prompt_enhancer/pe_worker.py", "tools/convert/qwen_image.py",
            # The port carries its own contract across, so the ported tree can re-verify itself at the
            # next upstream sync instead of trusting a person to remember the manifest.
            "tools/check_image_seams.py", "tools/image_lane_copy_set.py", "tools/port_lane.ps1",
            "docs/maintainer/image-integration-points.md", "docs/maintainer/image-lane.copy-set.json")
SKIP_DIRS = {".git", "build-windows", "build", "third_party", "node_modules", ".superpowers",
             ".agent", "out", "profiles", "__pycache__"}

TARGET_RE = re.compile(r"\badd_(?:executable|library)\s*\(\s*([A-Za-z0-9_][A-Za-z0-9_\-.]*)")
LIB_RE = re.compile(r"\binfernix_[a-z0-9_]+")
HELPER_RE = re.compile(r"\binfernix_add_[a-z_]*test\s*\(\s*([A-Za-z0-9_][A-Za-z0-9_\-.]*)")
HELPER_NAME_RE = re.compile(r"^infernix_add_[a-z_]*test$")
FILE_TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.(?:cpp|cu|c|cc|h|hpp|cuh)")
INC_ATTR_RE = re.compile(r'<(?:ClCompile|ClInclude|CudaCompile|Cuda|None|CustomBuild|Image)\s+Include="([^"]+)"')
DIRS_TAG_RE = re.compile(r"<AdditionalIncludeDirectories>(.*?)</AdditionalIncludeDirectories>", re.S)
INCLUDE_LINE_RE = re.compile(r"^\s*#\s*include\s*[\"<]([^\">]+)[\">]", re.M)


def git_files(repo, ref):
    out = subprocess.run(["git", "-C", repo, "ls-tree", "-r", "--name-only", ref],
                         capture_output=True, text=True, encoding="utf-8", check=True)
    return {ln.strip().replace("\\", "/") for ln in out.stdout.splitlines() if ln.strip()}


def git_resolve(repo, ref):
    out = subprocess.run(["git", "-C", repo, "rev-parse", "--verify", "--quiet", ref + "^{commit}"],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0 or not out.stdout.strip():
        raise SystemExit("upstream ref %r is not present locally; fetch it first" % ref)
    return out.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--build-dir", default="build-windows")
    ap.add_argument("--write-json", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    build = os.path.abspath(args.build_dir)
    if not os.path.isdir(build):
        raise SystemExit("no build directory at %s: configure it first -- the copy set is derived "
                         "from CMake's generated projects, not guessed" % build)

    upstream_sha = git_resolve(repo, args.upstream)
    upstream = git_files(repo, upstream_sha)
    ours_head = git_files(repo, "HEAD")

    present = set(ours_head)
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not name.endswith(".pyc"):
                present.add(os.path.relpath(os.path.join(dirpath, name), repo).replace("\\", "/"))
    missing = present - upstream

    def is_file(rel):
        return rel in present and os.path.isfile(os.path.join(repo, rel.replace("/", os.sep)))

    cache = {}

    def read(rel):
        if rel not in cache:
            try:
                with open(os.path.join(repo, rel.replace("/", os.sep)), encoding="utf-8", errors="replace") as fh:
                    cache[rel] = fh.read()
            except OSError:
                cache[rel] = ""
        return cache[rel]

    seeds = sorted(f for f in missing if f.endswith(".cmake") and is_file(f))
    targets = set()
    for s in seeds:
        text = read(s)
        targets.update(TARGET_RE.findall(text))
        targets.update(LIB_RE.findall(text))
        targets.update(HELPER_RE.findall(text))
    targets = {t for t in targets
               if "/" not in t and not t.endswith((".cpp", ".cu", ".h", ".cmake", ".py"))
               and not HELPER_NAME_RE.match(t) and t != "infernix_internal_includes"}

    vindex = {}
    for dirpath, _dn, filenames in os.walk(build):
        for name in filenames:
            if name.endswith(".vcxproj"):
                vindex.setdefault(name[: -len(".vcxproj")], os.path.join(dirpath, name))

    rootfwd = repo.replace("\\", "/")
    per_target, incdirs, unknown = {}, set(), []
    for name in sorted(targets):
        path = vindex.get(name)
        if not path:
            unknown.append(name)
            continue
        # Generated item paths are relative to the PROJECT file's own directory, which lives under
        # build-windows (itself under the repo root): express it against the repo root, or the
        # "..\..\..\src\..." CUDA items land one level above the tree and silently vanish.
        rel_dir = posixpath.dirname(os.path.relpath(path, repo).replace("\\", "/"))
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        items = set()
        for m in INC_ATTR_RE.findall(text):
            q = m.replace("\\", "/")
            if len(q) > 1 and q[1] == ":":
                items.add(posixpath.relpath(q, rootfwd))
            else:
                items.add(posixpath.normpath(posixpath.join(rel_dir, q)))
        per_target[name] = items
        for blob in DIRS_TAG_RE.findall(text):
            for d in blob.split(";"):
                d = d.strip().replace("\\", "/")
                if not d or any(tok in d for tok in ("$(", "@(", "%(", "<<")):
                    continue
                if len(d) > 1 and d[1] == ":":
                    d = posixpath.relpath(d, rootfwd)
                elif not posixpath.isabs(d):
                    d = posixpath.normpath(posixpath.join(rel_dir, d))
                if posixpath.isdir(os.path.join(repo, d.replace("/", os.sep))):
                    incdirs.add(d)

    seed_text = "\n".join(read(s) for s in seeds)
    seed_names = {posixpath.basename(m) for m in FILE_TOKEN_RE.findall(seed_text)}
    files, lane_only, shared = set(), [], []
    for name, fs in sorted(per_target.items()):
        compiles = [f for f in fs if f.endswith(COMPILE_EXT)]
        if compiles and all(f in missing for f in compiles):
            lane_only.append(name)
            files |= fs
        else:
            shared.append(name)
            files |= {f for f in fs if posixpath.basename(f) in seed_names}

    dirs_sorted = sorted(incdirs)

    def resolve(inc, fromdir):
        for cand in [posixpath.normpath(posixpath.join(fromdir, inc))] + \
                    [posixpath.normpath(posixpath.join(d, inc)) for d in dirs_sorted]:
            if cand.startswith("../") or cand.startswith("/"):
                continue
            if is_file(cand):
                return cand
        return None

    work = [f for f in files if is_file(f)]
    seen = set(work)
    while work:
        f = work.pop()
        if not f.endswith(SOURCE_EXT):
            continue
        for inc in INCLUDE_LINE_RE.findall(read(f)):
            r = resolve(inc, posixpath.dirname(f))
            if r and r not in seen:
                seen.add(r)
                work.append(r)

    base = {f for f in seen if f in missing and is_file(f)} | {s for s in seeds if is_file(s)}
    lane_dirs = sorted({posixpath.dirname(f) for f in base})
    runtime = {f for f in missing if is_file(f) and f.endswith(RUNTIME_EXT)
               and any(f.startswith(d + "/") for d in lane_dirs)}
    chosen = {f for f in (base | runtime | {f for f in EXPLICIT if is_file(f)}) if f not in EXCLUSIONS}
    copy_set = sorted(chosen)

    doc = {
        "generated_by": "tools/image_lane_copy_set.py",
        "upstream": upstream_sha,
        "note": "Every path here exists in this tree and not at <upstream>: the port copies it. "
                "Regenerate with --write-json after adding or reaching a lane file; "
                "tools/check_image_seams.py fails the tree when this file goes stale.",
        "exclusions": EXCLUSIONS,
        "files": copy_set,
    }

    if args.write_json:
        target = os.path.join(repo, args.write_json.replace("/", os.sep))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=2, sort_keys=False)
            fh.write("\n")
        if not args.quiet:
            print("wrote      : %s" % args.write_json)

    if not args.quiet:
        print("seeds      : %d ours-only cmake file(s)" % len(seeds))
        print("targets    : %d resolved, %d entirely lane, %d shared, %d without a project file"
              % (len(targets), len(lane_only), len(shared), len(unknown)))
        if unknown:
            print("  no project: %s" % ", ".join(sorted(unknown)[:12]))
        print("copy set   : %d file(s) to carry across" % len(copy_set))
        print("excluded   : %d (each justified in the JSON)" % len(EXCLUSIONS))

    if args.check:
        json_path = args.write_json or "docs/maintainer/image-lane.copy-set.json"
        committed_path = os.path.join(repo, json_path.replace("/", os.sep))
        if not os.path.isfile(committed_path):
            print("COPYSET_MISSING: %s does not exist -- run with --write-json" % json_path)
            return 1
        with open(committed_path, encoding="utf-8") as fh:
            committed = json.load(fh)
        before = set(committed.get("files", []))
        now = set(copy_set)
        gone = sorted(f for f in before if not os.path.isfile(os.path.join(repo, f.replace("/", os.sep))))
        if gone:
            for f in gone:
                print("COPYSET_GONE: %s is listed but no longer exists" % f)
        if before != now:
            for f in sorted(before - now):
                print("COPYSET_STALE: %s is listed but no longer reached by the lane" % f)
            for f in sorted(now - before):
                print("COPYSET_STALE: %s is needed by the lane but not listed" % f)
        if gone or before != now:
            print("VERDICT: copy set is out of date -- rerun tools/image_lane_copy_set.py --write-json")
            return 1
        print("VERDICT: copy set is current (%d files)" % len(now))
    return 0


if __name__ == "__main__":
    sys.exit(main())
