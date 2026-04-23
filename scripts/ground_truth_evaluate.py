"""Ground-truth evaluation — score matcher verdicts against human labels.

Reads labels from ``tests/fixtures/ground_truth/{pair_stem}.md`` (the human
working document — chips + reasoning + labels live in one place) and joins
them to the matcher verdicts in ``tests/fixtures/ground_truth/{pair_stem}.csv``
by ``pair_id``.  Prints a per-matcher confusion matrix plus precision /
recall / F1.  Ambiguous rows are reported separately and excluded from the
scored denominators.

Usage
-----

  uv run python scripts/ground_truth_evaluate.py <scene_a> <scene_b>

where ``<scene_a>`` and ``<scene_b>`` are the same canonical scene names
used by ``scripts/ground_truth_prepare.py`` (chronological order).

Label format in the MD
----------------------

Each pair section contains a line of the form:

    **Human label:** `_____` (same / different / ambiguous)

To label a pair, replace ``_____`` between the backticks with one of
``same`` / ``different`` / ``ambiguous``.  Anything else (including the
literal ``_____`` placeholder) is treated as "unfilled" and skipped.

The CSV is read only for the matcher verdicts (``direct_accept``,
``signature_accept``); its ``human_label`` column is ignored.

Conventions
-----------

* Human label ``same`` is treated as **ground-truth positive** — the two
  observations refer to the same physical feature across the acquisition
  gap.
* Human label ``different`` is treated as **ground-truth negative**.
* Human label ``ambiguous`` is excluded from scoring (counted separately so
  the user can see how many pairs were too hard to call).
* A matcher "accepts" the pair iff its corresponding column (``direct_accept``
  or ``signature_accept``) is 1.  Accept = predicted positive.

The scored set is "all pairs within the 50 m spatial gate".  Pairs beyond
that gate are invisible to both matchers by construction and are not in the
sheet.

Confusion matrix cells
----------------------

For each matcher, on the labeled (non-ambiguous) rows:

  TP = accepted and same
  FP = accepted and different  (false accept)
  FN = rejected and same       (false reject)
  TN = rejected and different  (correct reject)

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * P * R / (P + R)

This script does NOT write any files — output is stdout only.  It does NOT
modify the CSV.  It is safe to run as often as the user iterates on labels.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ground_truth_prepare import SCENES, pair_stem  # noqa: E402


GROUND_TRUTH_DIR = REPO_ROOT / "tests/fixtures/ground_truth"

VALID_LABELS = {"same", "different", "ambiguous"}

# Matches "## pair_NN — ..." section headers.
_PAIR_HEADER_RE = re.compile(r"^##\s+(pair_\d+)\b", re.MULTILINE)
# Matches the label line inside a section.  Captures the content between the
# backticks after "**Human label:**".  Placeholder "_____" counts as unfilled.
_LABEL_RE = re.compile(r"\*\*Human label:\*\*\s*`([^`]*)`")


def _parse_labels_from_md(md_path: Path) -> dict[str, str]:
    """Return {pair_id: label_str} for every pair section in the MD.

    ``label_str`` is the raw content between the backticks (possibly empty,
    or the placeholder ``_____``).  Caller normalizes.
    """
    text = md_path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    headers = list(_PAIR_HEADER_RE.finditer(text))
    for idx, m in enumerate(headers):
        pair_id = m.group(1)
        start = m.start()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        section = text[start:end]
        lm = _LABEL_RE.search(section)
        out[pair_id] = (lm.group(1) if lm else "").strip()
    return out


def _classify(accepted: bool, label: str) -> str:
    if label == "same":
        return "TP" if accepted else "FN"
    if label == "different":
        return "FP" if accepted else "TN"
    raise ValueError(f"unexpected label for scored row: {label!r}")


def _score(counts: dict[str, int]) -> tuple[float, float, float]:
    tp = counts["TP"]
    fp = counts["FP"]
    fn = counts["FN"]
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * p * r / (p + r) if (p + r) else float("nan")
    return p, r, f1


def _summarize_matcher(name: str, counts: dict[str, int]) -> None:
    p, r, f1 = _score(counts)
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]
    n_pos = tp + fn
    n_neg = fp + tn
    print(f"  {name}")
    print(f"    confusion (scored rows only):")
    print(f"      TP={tp:3d}  FP={fp:3d}  (accepts: correct + false)")
    print(f"      FN={fn:3d}  TN={tn:3d}  (rejects: false  + correct)")
    print(f"    precision = {p:.3f}   recall = {r:.3f}   f1 = {f1:.3f}")
    print(f"    positives in scored set: {n_pos}   negatives: {n_neg}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score matcher verdicts against hand-labeled ground truth.",
    )
    p.add_argument("scene_a", choices=sorted(SCENES), help="Earlier scene (A).")
    p.add_argument("scene_b", choices=sorted(SCENES), help="Later scene (B).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    stem = pair_stem(args.scene_a, args.scene_b)
    sheet_csv = GROUND_TRUTH_DIR / f"{stem}.csv"
    sheet_md = GROUND_TRUTH_DIR / f"{stem}.md"

    for p, name in ((sheet_csv, "CSV"), (sheet_md, "MD")):
        if not p.exists():
            print(
                f"ERROR: {p} does not exist.  Run "
                f"`scripts/ground_truth_prepare.py {args.scene_a} {args.scene_b}` first."
            )
            sys.exit(1)

    with sheet_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    labels = _parse_labels_from_md(sheet_md)

    # Join MD labels into the CSV rows by pair_id; CSV's own human_label column
    # is intentionally ignored.
    for r in rows:
        raw = labels.get(r["pair_id"], "").strip().lower()
        if raw in {"", "_____"}:
            r["_label"] = ""       # unfilled
        else:
            r["_label"] = raw

    missing_from_md = [r["pair_id"] for r in rows if r["pair_id"] not in labels]
    n_total = len(rows)
    unfilled = [r for r in rows if not r["_label"]]
    ambiguous = [r for r in rows if r["_label"] == "ambiguous"]
    invalid = [r for r in rows if r["_label"] and r["_label"] not in VALID_LABELS]
    scored = [r for r in rows if r["_label"] in {"same", "different"}]

    print(f"CSV:   {sheet_csv}")
    print(f"MD:    {sheet_md}  (label source)")
    print(f"  total rows:       {n_total}")
    print(f"  unfilled rows:    {len(unfilled)}")
    print(f"  ambiguous rows:   {len(ambiguous)}")
    print(f"  invalid labels:   {len(invalid)}")
    print(f"  scored rows:      {len(scored)} (same + different)")
    if missing_from_md:
        print(f"  WARNING: {len(missing_from_md)} CSV pair_ids have no matching "
              f"section in the MD: {missing_from_md[:5]}...")
    if invalid:
        print("  WARNING: rows with invalid labels (expected same/different/ambiguous):")
        for r in invalid:
            print(f"    {r['pair_id']}: {r['_label']!r}")
    if unfilled:
        print("  NOTE: unfilled rows are skipped.")

    if not scored:
        print("\nNo scored rows — fill in `human_label` values and rerun.")
        return

    # Per-matcher confusion
    matchers = [
        ("DirectSpatialMatcher(gate_m=50)",              "direct_accept"),
        ("SignatureMatcher(gate_m=50, sig_gate=0.8)",    "signature_accept"),
    ]

    print("\nMatcher scores (scored rows only, ambiguous excluded):\n")
    for name, col in matchers:
        counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for r in scored:
            accepted = (r[col] or "").strip() in {"1", "true", "True"}
            counts[_classify(accepted, r["_label"])] += 1
        _summarize_matcher(name, counts)
        print()

    # Head-to-head diff — pairs where the two matchers disagree, by human label
    disagreement: dict[tuple[str, str], list[str]] = {}
    for r in scored:
        d_acc = (r["direct_accept"] or "").strip() in {"1", "true", "True"}
        s_acc = (r["signature_accept"] or "").strip() in {"1", "true", "True"}
        if d_acc == s_acc:
            continue
        key = (
            "Direct-only accept" if d_acc else "Signature-only accept",
            r["_label"],
        )
        disagreement.setdefault(key, []).append(r["pair_id"])

    if disagreement:
        print("Head-to-head disagreements on scored rows:")
        for (which, label), pair_ids in sorted(disagreement.items()):
            print(f"  {which}, human={label}: {len(pair_ids)} pairs "
                  f"[{', '.join(pair_ids)}]")
    else:
        print("No head-to-head disagreements on scored rows.")


if __name__ == "__main__":
    main()
