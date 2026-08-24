"""Tests for the Pedagogical Storybook & Visualizer (beads 2l0, 4x9)."""

from __future__ import annotations

from pathlib import Path

from scripts.bio_storybook import (
    generate_storybook_html,
    main as storybook_main,
    run_neuromod_chapter,
    run_presynaptic_chapter,
)


def test_storybook_chapters_generate_valid_traces():
    """Chapters simulate finite steps with valid biophysical quantities."""
    ch1 = run_presynaptic_chapter(steps=5)
    assert len(ch1.step_data) == 5
    assert ch1.step_data[0]["calcium"] >= 0.0
    assert ch1.step_data[0]["rrp_vesicles"] > 0.0
    assert ch1.step_data[0]["release_flux"] > 0.0

    ch2 = run_neuromod_chapter(steps=5)
    assert len(ch2.step_data) == 5
    assert ch2.step_data[0]["dopamine_da"] is not None
    assert ch2.step_data[0]["plasticity_gain"] > 0.0


def test_storybook_html_export(tmp_path: Path):
    """HTML storybook exports standalone HTML document with equations and tables."""
    target_html = tmp_path / "test_storybook.html"
    chapters = [run_presynaptic_chapter(steps=4), run_neuromod_chapter(steps=4)]

    out_file = generate_storybook_html(chapters, output_path=target_html)
    assert out_file.exists()

    content = out_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Presynaptic Biophysics" in content
    assert "Neuromodulation (DA / ACh / NE)" in content
    assert "<table>" in content


def test_storybook_cli_entrypoint(tmp_path: Path):
    """CLI entrypoint runs and exports HTML storybook."""
    target_html = tmp_path / "cli_storybook.html"
    ret = storybook_main(["--export-html", str(target_html)])
    assert ret == 0
    assert target_html.exists()
