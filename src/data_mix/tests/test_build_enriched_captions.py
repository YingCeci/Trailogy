"""Tests for ``build_enriched_captions.py``.

Focus areas:
  * ``build_enriched_answer`` — eval-anchor preservation, field
    inclusion / exclusion, ``None``-enrichment fallback.
  * Helpers (``_dedupe_common_names``, ``_truncate_distribution``) —
    pure functions with clear edge cases.
  * ``_rebuild_row`` — image / slug / species / family / user-side
    preserved, only assistant turn modified, hard length cap kicks in
    when the rebuilt content exceeds ``MAX_CONTENT_CHARS``.
  * End-to-end CLI smoke — full main() over 2 tiny JSONLs, asserts
    the report file's counts.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from data_mix.src import build_enriched_captions as bec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestDedupeCommonNames:
    def test_returns_empty_for_none_or_blank(self) -> None:
        assert bec._dedupe_common_names(None, "rose") == []
        assert bec._dedupe_common_names("", "rose") == []
        assert bec._dedupe_common_names("   ", "rose") == []

    def test_drops_exact_case_variant_of_primary(self) -> None:
        # The case-only contract: "Common Pitcher Plant" is dropped
        # when primary is "common pitcher plant". Hyphen / spacing
        # variants ("red-osier" vs "red osier") survive — see the
        # docstring on _dedupe_common_names for the rationale.
        names = (
            "Common Pitcher Plant; common pitcher plant; "
            "northern pitcher plant"
        )
        out = bec._dedupe_common_names(names, "common pitcher plant")
        assert "Common Pitcher Plant" not in out
        assert "common pitcher plant" not in out
        assert "northern pitcher plant" in out

    def test_keeps_hyphen_variants(self) -> None:
        # Documents the current contract — hyphen / no-hyphen variants
        # are NOT collapsed. Lock-in test: if a future implementation
        # adds hyphen normalisation it should also touch this test.
        names = "red-osier dogwood; red osier dogwood; American dogwood"
        out = bec._dedupe_common_names(names, "red osier dogwood")
        assert "red-osier dogwood" in out
        assert "American dogwood" in out

    def test_preserves_order_and_dedupes_within_list(self) -> None:
        out = bec._dedupe_common_names(
            "balsam; Balsam; Canada Balsam; balsam",
            primary="balsam fir",
        )
        # The first "balsam" wins; later case-variant repeats are dropped
        # but the order of distinct names is preserved.
        assert out == ["balsam", "Canada Balsam"]


class TestTruncateDistribution:
    def test_returns_none_for_empty(self) -> None:
        assert bec._truncate_distribution(None, 5) is None
        assert bec._truncate_distribution("", 5) is None
        assert bec._truncate_distribution(";  ;", 5) is None

    def test_no_truncation_when_under_cap(self) -> None:
        out = bec._truncate_distribution("Alaska; Maine; Vermont", 5)
        assert out == "Alaska; Maine; Vermont"

    def test_appends_remainder_marker_when_over_cap(self) -> None:
        regions = "; ".join(f"R{i}" for i in range(20))
        out = bec._truncate_distribution(regions, 5)
        # 5 kept regions + 1 "and N more regions" tail.
        parts = out.split("; ")
        assert len(parts) == 6
        assert parts[:5] == ["R0", "R1", "R2", "R3", "R4"]
        assert parts[5] == "and 15 more regions"


# ---------------------------------------------------------------------------
# build_enriched_answer
# ---------------------------------------------------------------------------

class TestBuildEnrichedAnswer:
    def test_eval_anchor_is_first_sentence(self) -> None:
        # The eval scorer's species extractor depends on this exact
        # leading phrase. Regression here silently breaks every plant
        # eval, so it has its own dedicated test.
        out = bec.build_enriched_answer(
            "balsam fir", "Abies balsamea",
            {
                "common_name": "balsam fir",
                "scientific_name": "Abies balsamea",
                "accepted_scientific_name": "Abies balsamea",
                "wikipedia_summary": "A North American fir.",
                "gbif_distribution": "Maine; Vermont",
            },
        )
        assert out.startswith("Looks like balsam fir to me. ")

    def test_falls_back_to_compact_when_enriched_is_none(self) -> None:
        out = bec.build_enriched_answer(
            "balsam fir", "Abies balsamea", enriched=None,
        )
        # Compact fallback drops the legacy "...is a plant species
        # found in North America." tail per the v2-enrich design.
        assert out == (
            "Looks like balsam fir to me. "
            "Abies balsamea, commonly called balsam fir."
        )
        assert "plant species found in North America" not in out

    def test_compact_fallback_skips_scientific_clause_when_unknown(
        self,
    ) -> None:
        out = bec.build_enriched_answer(
            "mystery plant", "(unknown)", enriched=None,
        )
        assert out == "Looks like mystery plant to me."

    def test_emits_accepted_name_only_when_different(self) -> None:
        # accepted == scientific → simple form
        same = bec.build_enriched_answer(
            "balsam fir", "Abies balsamea",
            {"scientific_name": "Abies balsamea",
             "accepted_scientific_name": "Abies balsamea"},
        )
        assert "Scientific name: Abies balsamea." in same
        assert "accepted:" not in same

        # accepted != scientific → flag both
        diff = bec.build_enriched_answer(
            "Bailey acacia", "Acacia baileyana",
            {"scientific_name": "Acacia baileyana",
             "accepted_scientific_name": "Racosperma baileyanum"},
        )
        assert (
            "Scientific name: Acacia baileyana "
            "(accepted: Racosperma baileyanum)."
        ) in diff

    def test_excludes_gbif_description_and_profile(self) -> None:
        # The two noisy fields the v2-enrich design explicitly drops:
        # gbif_description (Latin typification etc.) and gbif_profile
        # (multilingual JSON / repeated habitat tokens). If a future
        # refactor adds them back, this test catches it.
        out = bec.build_enriched_answer(
            "common dandelion", "Taraxacum officinale",
            {
                "scientific_name": "Taraxacum officinale",
                "accepted_scientific_name": "Taraxacum officinale",
                "wikipedia_summary": "A common dandelion.",
                "gbif_description":
                    "Note: Farwell noted a nomen ambiguum...",
                "gbif_profile":
                    'habitat: Terrestrial; habitat: terrestrial; '
                    'lifeForm: {"lifeForm":["Árvore"]}',
                "gbif_distribution": "North America (PRESENT)",
            },
        )
        assert "nomen ambiguum" not in out
        assert "Terrestrial" not in out
        assert "Árvore" not in out
        # Sanity: the kept-fields ARE in there.
        assert "Wikipedia" not in out  # field name not echoed
        assert "A common dandelion." in out
        assert "North America (PRESENT)" in out


# ---------------------------------------------------------------------------
# _rebuild_row
# ---------------------------------------------------------------------------

class TestRebuildRow:
    def _row(self, slug: str) -> dict:
        return {
            "image": f"/abs/path/{slug}/img.jpg",
            "slug": slug,
            "species": "Foo bar",
            "family": "Fooaceae",
            "conversations": [
                {"role": "user", "content": "What plant is this?"},
                {"role": "assistant",
                 "content": "Looks like foo to me. Foo bar, commonly "
                            "called foo, is a plant species found in "
                            "North America."},
            ],
        }

    def test_preserves_image_slug_species_family_and_user_turn(self) -> None:
        row = self._row("foo")
        enriched = {
            "foo": {
                "scientific_name": "Foo bar",
                "accepted_scientific_name": "Foo bar",
                "wikipedia_summary": "A foo.",
            }
        }
        new_row, had_enrich = bec._rebuild_row(row, enriched, Counter())
        assert had_enrich
        # Image / slug / species / family preserved verbatim.
        for k in ("image", "slug", "species", "family"):
            assert new_row[k] == row[k]
        # User turn preserved verbatim.
        assert new_row["conversations"][0] == row["conversations"][0]
        # Assistant turn modified.
        assert new_row["conversations"][1]["role"] == "assistant"
        assert new_row["conversations"][1]["content"] != (
            row["conversations"][1]["content"]
        )

    def test_falls_back_when_slug_missing_from_enrichment(self) -> None:
        row = self._row("unknown_slug")
        new_row, had_enrich = bec._rebuild_row(row, {}, Counter())
        assert not had_enrich
        # Still produces a valid assistant content.
        assert new_row["conversations"][1]["content"].startswith(
            "Looks like unknown slug to me."
        )

    def test_hard_caps_oversized_content(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Shrink the cap so we can verify the truncation path on a
        # short string. The production constant is 1200 chars; that
        # would require building a giant fake enrichment.
        monkeypatch.setattr(bec, "MAX_CONTENT_CHARS", 80)

        row = self._row("foo")
        enriched = {
            "foo": {
                "scientific_name": "Foo bar",
                "accepted_scientific_name": "Foo bar",
                # Wikipedia summary much longer than the 80-char cap.
                "wikipedia_summary": "x" * 500,
            }
        }
        truncated: Counter = Counter()
        new_row, _ = bec._rebuild_row(row, enriched, truncated)
        content = new_row["conversations"][1]["content"]
        assert len(content) <= 80
        assert content.endswith("…")
        # First sentence (eval anchor) survives the cap.
        assert content.startswith("Looks like foo to me.")
        assert truncated["foo"] == 1

    def test_rejects_row_without_assistant_turn(self) -> None:
        row = {
            "image": "/abs/foo.jpg", "slug": "foo",
            "conversations": [{"role": "user", "content": "?"}],
        }
        with pytest.raises(ValueError, match="missing conversations"):
            bec._rebuild_row(row, {}, Counter())


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------

def test_main_end_to_end_writes_jsonls_symlink_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "in"
    output_root = tmp_path / "out"
    images_dir = input_root / "images_resized"
    images_dir.mkdir(parents=True)
    (images_dir / "img1.jpg").write_bytes(b"fake")

    # Two-row train, one-row val. test split intentionally absent so
    # the "missing split" path is exercised too.
    train_rows = [
        {
            "image": str(images_dir / "img1.jpg"),
            "slug": "foo",
            "species": "Foo bar",
            "family": "Fooaceae",
            "conversations": [
                {"role": "user", "content": "What is this?"},
                {"role": "assistant",
                 "content": "Looks like foo to me. Foo bar, commonly "
                            "called foo, is a plant species found in "
                            "North America."},
            ],
        },
        {
            "image": str(images_dir / "img1.jpg"),
            "slug": "unenriched_slug",
            "species": "Bar baz",
            "family": "(unknown)",
            "conversations": [
                {"role": "user", "content": "And this?"},
                {"role": "assistant",
                 "content": "Looks like unenriched_slug to me."},
            ],
        },
    ]
    val_rows = [train_rows[0]]
    (input_root / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n"
    )
    (input_root / "val.jsonl").write_text(
        json.dumps(val_rows[0]) + "\n"
    )

    enriched_path = tmp_path / "enriched.jsonl"
    enriched_path.write_text(json.dumps({
        "slug": "foo",
        "common_name": "foo",
        "scientific_name": "Foo bar",
        "accepted_scientific_name": "Foo bar",
        "common_names": "Foo; Fooey",
        "wikipedia_summary": "Foo is a plant.",
        "gbif_distribution": "North America (PRESENT); Maine; Vermont",
    }) + "\n")

    monkeypatch.setattr(sys, "argv", [
        "build_enriched_captions.py",
        "--input-root", str(input_root),
        "--enriched", str(enriched_path),
        "--output-root", str(output_root),
        "--log-level", "WARNING",
    ])
    bec.main()

    # Both rebuilt JSONLs exist.
    assert (output_root / "train.jsonl").exists()
    assert (output_root / "val.jsonl").exists()
    # test split skipped (warning logged, no file written).
    assert not (output_root / "test.jsonl").exists()

    # Images symlinked, not copied.
    sym = output_root / "images_resized"
    assert sym.is_symlink()
    assert sym.resolve() == images_dir.resolve()

    # Build report records the per-split counts and missing-enrichment
    # slug.
    report = json.loads((output_root / "build_report.json").read_text())
    assert report["enriched_unique_slugs"] == 1
    assert report["splits"]["train"]["n_rows"] == 2
    assert report["splits"]["train"]["n_rows_missing_enrichment"] == 1
    assert report["splits"]["train"]["missing_slugs_sample"] == [
        "unenriched_slug"
    ]
    assert "test" not in report["splits"]

    # Spot-check the rebuilt foo row: anchor preserved, wiki summary
    # included, GBIF distribution included.
    train_out = [
        json.loads(l)
        for l in (output_root / "train.jsonl").read_text().splitlines()
        if l.strip()
    ]
    foo_row = next(r for r in train_out if r["slug"] == "foo")
    content = foo_row["conversations"][1]["content"]
    assert content.startswith("Looks like foo to me.")
    assert "Foo is a plant." in content
    assert "North America (PRESENT); Maine; Vermont" in content
    # Old "is a plant species found in North America." tail removed.
    assert "is a plant species found in North America" not in content
