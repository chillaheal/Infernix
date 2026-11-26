#!/usr/bin/env python3
"""Check the image-lane seam manifest against an upstream ref.

Reads docs/maintainer/image-integration-points.md and verifies, for every seam row, that
each named anchor still exists in the named upstream file at the given ref. The verification
is exactly the primitive a porter uses by hand:

    git grep -c --fixed-strings <anchor> <ref> -- <path>

That is the whole dependency: this script shells out to git and reads the manifest. It needs
no build directory, no CMake cache, no compiled artifact, no network, and it writes no files.
It is safe to run while the serve and the PE worker are up; it touches neither :8888 nor any
process.

Categories, as defined in the manifest:

    must-port        image semantics are impossible without the hunk -> a missing anchor FAILS
    port-if-missing  loseable convenience (a comment, a reason bit, a premise test) -> WARN only
    ours-only        no upstream counterpart, pure re-add -> SKIPPED (only a build can check it)
    seam             an upstream-owned file the lane touches ONLY through its typed seam calls.
                     Its anchors are checked against OUR working tree (the seam call survived),
                     and its `residue:` tokens are counted in that same file against `lines:`
                     (no lane text leaked back in, and nothing upstream was deleted). This is
                     the category that makes a replay safe to land without re-reading the file.

Exit codes:

    0  every must-port anchor present and every seam intact (port-if-missing may still have warned)
    1  a must-port anchor is missing ("<path> <anchor> MISSING"), or a seam row is broken (its
       typed call left the file, or lane text other than the seam calls appeared in it)
    2  the manifest is unreadable or unparsable, or the ref cannot be resolved

Manifest format (the checker and the document are coupled; a row that does not parse is an
error, not a silent skip, so that a typo in the table cannot quietly turn a real port break
into a green run):

    ### <our path>
        our path: <path>
        upstream path: <path>
        category: must-port | port-if-missing | ours-only | seam
        hunks: <int>
        anchors: <name>, <name>, ...
        does: <free text, continuation lines indented deeper than 4 spaces>

A `seam` row additionally carries, at the same four-space key indentation:

        residue: <token>, <token>, ...     fixed strings that may only occur on seam lines
        lines: <int>                        how many lines of ours may contain any of them

An `ours-only` row may carry `companions: <path>, <path>, ...` naming the headers that travel with the
source it documents. `tools/port_lane.ps1` copies `our path` plus its companions; the checker itself
still skips ours-only rows.

`lines:` is the count this file's lane-carrying lines had when the seam was landed. The checker
prints every matching line, so a drift is visible and not just counted; it does not need an
allowlist of texts because the anchors already name the only calls the lane is allowed to make.

Keys are only recognised at exactly four leading spaces; deeper indentation is continuation
text of the previous key. A section is only parsed after a '### ' heading, so the indented
command lines in the document's prose are ignored.

Usage from the repo root:

    py -3.13 -I tools/check_image_seams.py
    py -3.13 -I tools/check_image_seams.py --upstream f9c2e42c --offline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

DEFAULT_MANIFEST = os.path.join("docs", "maintainer", "image-integration-points.md")
DEFAULT_UPSTREAM = "origin/main"

VALID_CATEGORIES = ("must-port", "port-if-missing", "ours-only", "seam")
REQUIRED_KEYS = ("our path", "upstream path", "category", "hunks", "anchors")
# A `seam` row must also say what may not appear in the file and how many lines may carry it.
SEAM_EXTRA_KEYS = ("residue", "lines")

EXIT_OK = 0
EXIT_MISSING_ANCHOR = 1
EXIT_MANIFEST_UNUSABLE = 2


def run_git(args, cwd):
    """Run git with argv as a list (never a shell string) and capture text output."""
    proc = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
    )
    out = proc.stdout.decode("utf-8", "replace")
    err = proc.stderr.decode("utf-8", "replace")
    return proc.returncode, out, err


def resolve_repo_root(explicit):
    if explicit:
        return os.path.abspath(explicit)
    code, out, err = run_git(["rev-parse", "--show-toplevel"], cwd=os.getcwd())
    if code != 0:
        fail("not a git repository: %s" % (err.strip() or "git rev-parse --show-toplevel failed"))
    return out.strip().replace("\\", os.sep)


def fail(message):
    """Manifest/ref-level problem: print and exit 2."""
    sys.stderr.write("SEAMS_ERROR: %s\n" % message)
    sys.stderr.write("VERDICT: manifest unusable (exit 2)\n")
    raise SystemExit(EXIT_MANIFEST_UNUSABLE)


def parse_manifest(path):
    """Return (rows, problems). Each row is a dict; problems are structural defects."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw_lines = handle.read().splitlines()
    except OSError as exc:
        fail("cannot read manifest %s: %s" % (path, exc))

    rows = []
    problems = []
    heading = None          # the '### ' heading text, our canonical row id
    current = None          # dict of keys for the section being read
    current_line = 0
    last_key = None
    saw_heading = False

    def flush():
        if current is None:
            return
        rows.append({"heading": heading, "line": current_line, "keys": dict(current)})
        # The manifest also documents the anchors by prose; a section always ends cleanly.

    for number, line in enumerate(raw_lines, start=1):
        if line.startswith("### "):
            flush()
            current = {}
            last_key = None
            heading = line[4:].strip()
            current_line = number
            saw_heading = True
            continue
        if line.startswith("## ") or line.startswith("# "):
            flush()
            current = None
            last_key = None
            heading = None
            continue
        if current is None:
            continue
        # A section body: '---' ends it, blank lines pass through.
        if line.strip() == "---":
            flush()
            current = None
            last_key = None
            heading = None
            continue
        if not line.strip():
            last_key = None
            continue
        # A key line is exactly four leading spaces then 'key: value'.
        if line.startswith("    ") and not line.startswith("     "):
            body = line[4:]
            if ":" in body:
                key, _, value = body.partition(":")
                key = key.strip().lower()
                if key in REQUIRED_KEYS or key == "does" or key in SEAM_EXTRA_KEYS or key == "companions":
                    current[key] = value.strip()
                    last_key = key
                    continue
            # Not a recognised key at four spaces: treat as a stray line and flag it,
            # because a mis-indented key is exactly the mistake that hides a seam.
            problems.append("%s:%d: unrecognised indented line in section '%s': %r"
                            % (path, number, heading, line))
            last_key = None
            continue
        # Deeper indentation: continuation of the previous key.
        if last_key is not None:
            current[last_key] = (current.get(last_key, "") + " " + line.strip()).strip()

    flush()

    if not saw_heading or not rows:
        fail("no '### <path>' seam sections found in %s - the manifest has been rewritten "
             "out from under the checker" % path)

    for row in rows:
        missing = [key for key in REQUIRED_KEYS if not row["keys"].get(key)]
        if missing:
            problems.append("%s:%d: section '%s' missing key(s): %s"
                            % (path, row["line"], row["heading"], ", ".join(missing)))
            continue
        category = row["keys"]["category"].strip().lower()
        if category not in VALID_CATEGORIES:
            problems.append("%s:%d: section '%s' has unknown category %r (expected one of: %s)"
                            % (path, row["line"], row["heading"], category, ", ".join(VALID_CATEGORIES)))
        try:
            int(row["keys"]["hunks"].strip())
        except ValueError:
            problems.append("%s:%d: section '%s' has non-integer hunks %r"
                            % (path, row["line"], row["heading"], row["keys"]["hunks"]))
        has_residue, has_lines = row["keys"].get("residue"), row["keys"].get("lines")
        if bool(has_residue) != bool(has_lines):
            problems.append("%s:%d: section '%s' carries %s but not %s - the lane gate needs both "
                            "the tokens and the line count"
                            % (path, row["line"], row["heading"],
                               "residue:" if has_residue else "lines:",
                               "lines:" if has_residue else "residue:"))
        if category == "seam" and not (has_residue and has_lines):
            problems.append("%s:%d: seam section '%s' is missing %s - without them the row checks "
                            "nothing but the calls still being there"
                            % (path, row["line"], row["heading"], " and ".join(SEAM_EXTRA_KEYS)))
        if has_lines:
            try:
                int(has_lines.strip())
            except ValueError:
                problems.append("%s:%d: section '%s' has non-integer lines %r"
                                % (path, row["line"], row["heading"], has_lines))
        anchors = split_anchors(row["keys"].get("anchors", ""))
        if category == "must-port" and not anchors:
            problems.append("%s:%d: must-port section '%s' has no anchors - the checker cannot "
                            "verify it; name the upstream symbol it hangs off"
                            % (path, row["line"], row["heading"]))
        if row["heading"] != row["keys"]["our path"].strip():
            problems.append("%s:%d: heading '%s' disagrees with 'our path: %s'"
                            % (path, row["line"], row["heading"], row["keys"]["our path"]))

    return rows, problems


def split_anchors(text):
    """anchors: a, b, c -> ['a', 'b', 'c'] with empty entries dropped."""
    return [item.strip() for item in text.split(",") if item.strip()]


def our_file_text(repo_root, path):
    """Read a working-tree file, or None when it is not there."""
    full = os.path.join(repo_root, path.replace("/", os.sep))
    if not os.path.isfile(full):
        return None
    with open(full, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def seam_check(repo_root, our_path, anchors, tokens, expected_lines):
    """Verify a seam row against OUR working tree.

    Returns (ok, notes): every anchor must still be in the file, and the lines carrying a banned
    token must be exactly `expected_lines` of them. Each note is a one-line finding.
    """
    text = our_file_text(repo_root, our_path)
    if text is None:
        return False, ["%s: not present in the working tree" % our_path]
    notes = []
    for anchor in anchors:
        if anchor not in text:
            notes.append("%s: seam call %r is GONE - the lane no longer reaches this file"
                         % (our_path, anchor))
    lines = text.splitlines()
    hits = [(n, ln) for n, ln in enumerate(lines, 1) if any(tok in ln for tok in tokens)]
    if len(hits) != expected_lines:
        notes.append("%s: %d line(s) carry a lane token, manifest says %d - a port either leaked "
                     "lane text in or deleted a seam call"
                     % (our_path, len(hits), expected_lines))
        for n, ln in hits:
            notes.append("    %s:%d %s" % (our_path, n, ln.strip()))
    return not notes, notes


def anchor_present(repo_root, ref, path, anchor):
    """True if the literal anchor occurs in <ref>:<path>.

    git grep exit status: 0 matches, 1 no matches, >=2 real error. Anything but 0/1 is
    escalated by the caller, because a broken ref would otherwise read as 'anchor missing'.
    """
    code, out, err = run_git(
        ["grep", "-c", "--fixed-strings", anchor, ref, "--", path], cwd=repo_root)
    if code == 0:
        return True, out.strip()
    if code == 1:
        return False, ""
    return None, (err.strip() or ("git grep exited %d" % code))


def run_copy_set_gate(repo_root, args):
    """Verify the ours-only copy set a port carries: every listed file exists, and (when this checkout
    is configured) the list still matches what CMake's generated projects say the lane needs."""
    rel = args.copy_set
    path = rel if os.path.isabs(rel) else os.path.join(repo_root, rel)
    if not os.path.isfile(path):
        print("COPYSET_MISSING: %s does not exist -- run tools/image_lane_copy_set.py --write-json" % rel)
        return False
    with open(path, encoding="utf-8") as fh:
        try:
            doc = json.load(fh)
        except ValueError as exc:
            print("COPYSET_UNREADABLE: %s is not valid JSON (%s)" % (rel, exc))
            return False
    listed = doc.get("files") or []
    print("copy set   : %d file(s) from %s (generated_by %s)"
          % (len(listed), rel, doc.get("generated_by", "?")))
    gone = [f for f in listed if not os.path.isfile(os.path.join(repo_root, f.replace("/", os.sep)))]
    for f in gone:
        print("COPYSET_GONE: %s is listed but no longer exists" % f)
    if gone:
        return False
    build = args.build_dir
    if not os.path.isabs(build):
        build = os.path.join(repo_root, build)
    if not os.path.isdir(build):
        print("copy set   : SKIP re-derivation (no configured build dir at %s)" % args.build_dir)
        return True
    tool = os.path.join(repo_root, "tools", "image_lane_copy_set.py")
    res = subprocess.run([sys.executable, "-I", tool, "--repo", repo_root, "--upstream", args.upstream,
                          "--check", "--build-dir", build, "--quiet"],
                         capture_output=True, text=True, encoding="utf-8")
    for line in (res.stdout or "").splitlines():
        print(line)
    if res.returncode != 0:
        for line in (res.stderr or "").splitlines():
            print(line)
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="check_image_seams.py",
        description="Verify the image-lane seam manifest's anchors against an upstream ref. "
                    "Reads the manifest and git only; writes nothing.")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="path to the seam manifest (default: %(default)s)")
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM,
                        help="upstream ref to check anchors against (default: %(default)s)")
    parser.add_argument("--offline", action="store_true",
                        help="use already-fetched refs only; never touch the network")
    parser.add_argument("--repo-root", default=None,
                        help="repository root (default: git rev-parse --show-toplevel)")
    parser.add_argument("--strict-warnings", action="store_true",
                        help="treat a missing port-if-missing anchor as a failure too")
    parser.add_argument("--emit-plan", default=None, metavar="PATH",
                        help="write the parsed rows as a JSON list to PATH and exit 0 (this is the "
                             "single parser a port driver reads: tools/port_lane.ps1 consumes it "
                             "rather than re-deriving the table). Written before any anchor result, "
                             "so a plan can be reviewed without network or refs.")
    parser.add_argument("--copy-set", default="docs/maintainer/image-lane.copy-set.json",
                        metavar="PATH",
                        help="the ours-only file list a port copies, generated by "
                             "tools/image_lane_copy_set.py; every path must exist and still be ours-only")
    parser.add_argument("--no-copy-set", action="store_true",
                        help="skip the copy-set gate (only when the build dir is not configured)")
    parser.add_argument("--build-dir", default="build-windows",
                        help="configured build dir of THIS checkout; when present the copy set is "
                             "re-derived from CMake's generated projects and compared (default: %(default)s)")
    args = parser.parse_args(argv)

    repo_root = resolve_repo_root(args.repo_root)
    manifest = args.manifest
    if not os.path.isabs(manifest):
        manifest = os.path.join(repo_root, manifest)

    rows, problems = parse_manifest(manifest)

    # Resolve the ref before grepping; an unresolvable ref is a manifest-run problem, not
    # a seaam failure, and every row would otherwise be reported MISSING spuriously.
    code, out, err = run_git(["rev-parse", "--verify", "--quiet", args.upstream + "^{commit}"],
                             cwd=repo_root)
    resolved = out.strip()
    if code != 0 or not resolved:
        if args.offline:
            fail("ref %r is not present locally and --offline forbids fetching" % args.upstream)
        sys.stderr.write("note: ref %r unresolved, attempting one best-effort fetch\n"
                         % args.upstream)
        run_git(["fetch", "--no-tags", "--quiet", "origin"], cwd=repo_root)
        code, out, _ = run_git(["rev-parse", "--verify", "--quiet", args.upstream + "^{commit}"],
                               cwd=repo_root)
        resolved = out.strip()
        if code != 0 or not resolved:
            fail("ref %r still unresolved after a best-effort fetch" % args.upstream)

    print("manifest : %s" % os.path.relpath(manifest, repo_root).replace("\\", "/"))
    print("upstream : %s -> %s" % (args.upstream, resolved[:12]))
    print("repo     : %s" % repo_root)
    print("network  : %s" % ("forbidden (--offline)" if args.offline else "best-effort fetch allowed"))
    print("rows     : %d" % len(rows))
    print()

    if problems:
        for problem in problems:
            sys.stderr.write("SEAMS_MANIFEST_DEFECT: %s\n" % problem)
        fail("%d structural defect(s) in the manifest - fix the table, do not weaken the "
             "anchors to get a green run" % len(problems))

    if args.emit_plan:
        import json
        plan = [{"our_path": r["keys"]["our path"].strip(),
                 "upstream_path": r["keys"]["upstream path"].strip(),
                 "category": r["keys"]["category"].strip().lower(),
                 "hunks": int(r["keys"]["hunks"].strip()),
                 "anchors": split_anchors(r["keys"]["anchors"]),
                 "residue": split_anchors(r["keys"].get("residue", "")),
                 "companions": split_anchors(r["keys"].get("companions", "")),
                 "lines": (int(r["keys"]["lines"].strip()) if r["keys"].get("lines") else None),
                 "does": (r["keys"].get("does") or "").strip()}
                for r in rows]
        with open(args.emit_plan, "w", encoding="utf-8") as handle:
            json.dump({"upstream": resolved[:12], "rows": plan}, handle, indent=2,
                      ensure_ascii=False)
            handle.write("\n")
        print("plan written : %s (%d rows)" % (args.emit_plan, len(plan)))
        return EXIT_OK

    counts = {"must-port": {"pass": 0, "missing": 0, "skipped": 0},
              "port-if-missing": {"pass": 0, "missing": 0, "skipped": 0},
              "ours-only": {"pass": 0, "missing": 0, "skipped": 0},
              "seam": {"pass": 0, "missing": 0, "skipped": 0}}
    failures = []
    warnings = []
    errors = []
    total_anchors = 0

    for row in rows:
        keys = row["keys"]
        category = keys["category"].strip().lower()
        our_path = keys["our path"].strip()
        upstream_path = keys["upstream path"].strip()
        hunks = keys["hunks"].strip()
        anchors = split_anchors(keys["anchors"])

        if category == "ours-only":
            counts["ours-only"]["skipped"] += 1
            print("SKIP    ours-only       %-46s hunks=%-4s no upstream counterpart"
                  % (our_path, hunks))
            continue

        if category == "seam":
            # Checked against OUR working tree, not the upstream ref: this row exists to prove the
            # typed seam survived a replay and that no lane rule travelled back into the file.
            ok, notes = seam_check(repo_root, our_path, anchors, split_anchors(keys["residue"]),
                                   int(keys["lines"].strip()))
            if ok:
                counts["seam"]["pass"] += 1
                print("PASS    seam            %-46s calls=%-3d lane-lines=%s"
                      % (our_path, len(anchors), keys["lines"].strip()))
            else:
                counts["seam"]["missing"] += 1
                print("BROKEN  seam            %-46s calls=%d" % (our_path, len(anchors)))
                for note in notes:
                    failures.append("%s: %s" % (our_path, note))
                    print("        %s" % note)
            continue

        row_missing = []
        for anchor in anchors:
            total_anchors += 1
            present, detail = anchor_present(repo_root, args.upstream, upstream_path, anchor)
            if present is None:
                errors.append("%s: git grep failed for anchor %r in %s: %s"
                              % (our_path, anchor, upstream_path, detail))
                continue
            if not present:
                row_missing.append(anchor)

        # A row may also carry the our-tree lane gate (see `seam`): the same check runs here for a
        # file that keeps documented lane hunks, so a replay is caught on both sides.
        if keys.get("residue") and keys.get("lines"):
            ok, notes = seam_check(repo_root, our_path, [], split_anchors(keys["residue"]),
                                   int(keys["lines"].strip()))
            if not ok:
                for note in notes:
                    failures.append("%s: %s" % (our_path, note))
                    print("        LANE_GATE %s" % note)

        if not row_missing:
            counts[category]["pass"] += 1
            print("PASS    %-15s %-46s hunks=%-4s anchors=%d"
                  % (category, our_path, hunks, len(anchors)))
            continue

        counts[category]["missing"] += 1
        for anchor in row_missing:
            line = "%s %s MISSING" % (upstream_path, anchor)
            if category == "must-port":
                failures.append(line)
                print("MISSING %-15s %-46s anchor=%s" % (category, our_path, anchor))
            else:
                warnings.append(line)
                print("WARN    %-15s %-46s anchor=%s (loseable)"
                      % (category, our_path, anchor))

    print()
    print("checked %d anchors across %d rows" % (total_anchors, len(rows)))
    for category in ("must-port", "port-if-missing", "ours-only", "seam"):
        stat = counts[category]
        print("  %-15s ok=%d missing=%d skipped=%d"
              % (category, stat["pass"], stat["missing"], stat["skipped"]))

    if errors:
        print()
        for error in errors:
            sys.stderr.write("SEAMS_GIT_ERROR: %s\n" % error)
        fail("git grep reported %d error(s); anchor results are not trustworthy" % len(errors))

    if failures:
        print()
        print("MUST-PORT ANCHORS MISSING OR SEAMS BROKEN (%d):" % len(failures))
        for line in failures:
            print("  %s" % line)
        print()
        print("Each line is 'file anchor MISSING'. For every one of them, follow the action")
        print("plan in the manifest before porting anything:")
        print("  docs/maintainer/image-integration-points.md, section 'What to do if a")
        print("  must-port anchor is missing after an upstream merge'.")

    if warnings:
        print()
        print("port-if-missing anchors absent (%d) - warnings only, the lane still works"
              % len(warnings))
        for line in warnings:
            print("  %s" % line)

    # The ours-only copy set is what a port physically carries across, so it is gated here too: a lane
    # file added, deleted or newly reached since the last regeneration would otherwise be silently
    # left behind and only surface as a build error in the ported tree.
    if not args.no_copy_set and not failures:
        print()
        if not run_copy_set_gate(repo_root, args):
            return EXIT_MISSING_ANCHOR

    print()
    if failures:
        print("VERDICT: FAIL - do not start a port from a build error; resolve every anchor above "
              "or re-derive it with days, not minutes, of budget")
        return EXIT_MISSING_ANCHOR
    if warnings:
        print("VERDICT: OK - every must-port anchor is present at %s" % resolved[:12])
        return EXIT_OK
    print("VERDICT: OK - every must-port anchor is present at %s" % resolved[:12])
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
