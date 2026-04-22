"""Tests for the label-preservation logic in ``scripts/ground_truth_prepare.py``.

The ground-truth MD is the canonical record of human labels.  Re-running
``ground_truth_prepare.py`` against an already-labeled MD must preserve the
labels (otherwise the script silently destroys the labeling work).  These
tests guard the merge logic.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/ground_truth_prepare.py"


@pytest.fixture(scope="module")
def gt_module():
    """Load ``scripts/ground_truth_prepare.py`` as a module without side effects.

    The script is a CLI; importing the file via ``importlib`` runs only the
    module body, which defines the helpers we want to test (``main()`` is
    guarded by ``if __name__ == '__main__':``).  We isolate the module import
    so it does not pollute ``sys.modules`` for other tests.
    """
    spec = importlib.util.spec_from_file_location("ground_truth_prepare", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ground_truth_prepare"] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop("ground_truth_prepare", None)


def _make_section(pair_id: str, label: str) -> str:
    """Minimal pair section with only the fields the merge logic touches."""
    return dedent(f"""\
        ## {pair_id} — spatial 1.00 m, signature 0.500

        - **Direct:** ACCEPT   |   **Signature (V1):** ACCEPT

        **Human label:** `{label}` (same / different / ambiguous)

        ---

        """)


def _build_md(*sections: tuple[str, str]) -> str:
    head = "# header\n\n"
    return head + "".join(_make_section(pid, lbl) for pid, lbl in sections)


def test_parse_existing_labels_extracts_each_pair(gt_module, tmp_path):
    md = _build_md(
        ("pair_01", "same"),
        ("pair_02", "different"),
        ("pair_03", "ambiguous"),
        ("pair_04", "_____"),
    )
    p = tmp_path / "sheet.md"
    p.write_text(md, encoding="utf-8")
    labels = gt_module.parse_existing_labels(p)
    assert labels == {
        "pair_01": "same",
        "pair_02": "different",
        "pair_03": "ambiguous",
        "pair_04": "_____",
    }


def test_parse_existing_labels_missing_file_returns_empty(gt_module, tmp_path):
    assert gt_module.parse_existing_labels(tmp_path / "nope.md") == {}


def test_merge_labels_preserves_existing_and_blanks_new_pairs(gt_module):
    # Freshly-generated MD has 3 pairs, all with the placeholder.
    new_md = _build_md(
        ("pair_01", "_____"),
        ("pair_02", "_____"),
        ("pair_03", "_____"),  # new pair since last labeling
    )
    # Old labels: pair_01 was labeled "same", pair_02 was "ambiguous".
    # pair_03 is new and absent.
    old = {"pair_01": "same", "pair_02": "ambiguous"}
    merged, n_preserved, n_blank = gt_module.merge_labels(new_md, old)

    assert n_preserved == 2
    assert n_blank == 1
    # Preserved labels appear verbatim in the merged output.
    assert "**Human label:** `same` (same / different / ambiguous)" in merged
    assert "**Human label:** `ambiguous` (same / different / ambiguous)" in merged
    # The new pair retains the placeholder.
    assert merged.count("`_____`") == 1


def test_merge_labels_treats_blank_old_label_as_unfilled(gt_module):
    """An existing pair labeled with the placeholder shouldn't count as preserved."""
    new_md = _build_md(("pair_01", "_____"))
    merged, n_preserved, n_blank = gt_module.merge_labels(
        new_md, {"pair_01": "_____"}
    )
    assert n_preserved == 0
    assert n_blank == 1
    assert "`_____`" in merged


def test_merge_labels_empty_old_is_no_op(gt_module):
    new_md = _build_md(("pair_01", "_____"), ("pair_02", "_____"))
    merged, n_preserved, n_blank = gt_module.merge_labels(new_md, {})
    assert n_preserved == 0
    assert n_blank == 2
    # Output identical to input — no labels to splice.
    assert merged == new_md


def test_parse_existing_header_returns_html_comment(gt_module, tmp_path):
    md = "<!-- provenance line -->\n# header\n\n"
    p = tmp_path / "sheet.md"
    p.write_text(md, encoding="utf-8")
    assert gt_module.parse_existing_header(p) == "<!-- provenance line -->\n"


def test_parse_existing_header_no_comment_returns_empty(gt_module, tmp_path):
    p = tmp_path / "sheet.md"
    p.write_text("# header\n\nbody\n", encoding="utf-8")
    assert gt_module.parse_existing_header(p) == ""
