from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pytest
from jinja2 import Template

TEMPLATE_PATH = Path(__file__).parents[1] / "src" / "gsMap" / "report" / "static" / "template.html"


class _TemplateAuditParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.label_targets: list[str] = []
        self.control_targets: list[str] = []
        self.report_card_headers: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if element_id := attributes.get("id"):
            self.ids.append(element_id)
        if tag == "label" and (label_target := attributes.get("for")):
            self.label_targets.append(label_target)
        if control_target := attributes.get("aria-controls"):
            self.control_targets.append(control_target)
        if tag == "button" and "report-card-header" in (attributes.get("class") or ""):
            self.report_card_headers.append(attributes)


@pytest.fixture
def report_template() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


@pytest.fixture
def rendered_report(report_template: str) -> str:
    return Template(report_template).render(
        title="gsMap report test",
        project_name="test_project",
        generated_at="2026-08-17 00:00:00",
        gsmap_version="test",
    )


def test_report_template_has_unique_ids_and_valid_control_targets(
    rendered_report: str,
) -> None:
    parser = _TemplateAuditParser()
    parser.feed(rendered_report)

    counts = Counter(parser.ids)
    assert [element_id for element_id, count in counts.items() if count > 1] == []
    assert set(parser.control_targets) <= set(parser.ids)
    assert set(parser.label_targets) <= set(parser.ids)


def test_report_sections_are_accessible_buttons(rendered_report: str) -> None:
    parser = _TemplateAuditParser()
    parser.feed(rendered_report)

    assert len(parser.report_card_headers) == 7
    assert all(header.get("type") == "button" for header in parser.report_card_headers)
    assert all(header.get(":aria-expanded") for header in parser.report_card_headers)
    assert all(header.get("aria-controls") for header in parser.report_card_headers)


def test_report_template_includes_responsive_navigation_and_error_state(
    report_template: str,
) -> None:
    assert 'id="report-sidebar"' in report_template
    assert 'class="mobile-sidebar-backdrop' in report_template
    assert 'id="main-content"' in report_template
    assert 'class="report-overview' in report_template
    assert "initializationError" in report_template
    assert "setupSectionObserver()" in report_template
    assert "scrollToSection(section)" in report_template
    assert "setAllSections(false)" in report_template
    assert 'x-init="init()"' not in report_template


def test_report_plot_layout_has_no_fixed_minimum_panel_widths(
    report_template: str,
) -> None:
    assert "min-w-[400px]" not in report_template
    assert "min-w-[500px]" not in report_template
    assert "height: 800px" not in report_template
    assert report_template.count('class="plot-grid"') == 5


def test_removed_hidden_annotation_section_does_not_regress(report_template: str) -> None:
    assert 'id="sec-annotation"' not in report_template
    assert "If this message persists" not in report_template


def test_spatial3d_is_lazy_direct_plotly_and_labels_nonphysical_geometry(
    report_template: str,
) -> None:
    assert "<iframe" not in report_template
    assert "initializeSpatial3d()" in report_template
    assert "GSMAP_SPATIAL_3D_INDEX" in report_template
    assert 'id="spatial3dPlotTrait"' in report_template
    assert 'id="spatial3dPlotAnnotation"' in report_template
    assert "Expanded Z (visual aid)" in report_template
    assert "spatial3dCoordinateInfo?.note" in report_template
    assert "xRange / largestXyRange" in report_template
    assert "yRange / largestXyRange" in report_template


def test_large_secondary_datasets_are_loaded_near_their_sections(
    report_template: str,
) -> None:
    assert '<script src="js_data/cauchy_results.js"></script>' not in report_template
    assert '<script src="js_data/umap_data.js"></script>' not in report_template
    assert "setupDeferredPlotLoading()" in report_template
    assert "loadUmapData()" in report_template
    assert "loadCauchyData()" in report_template
    assert 'loading="lazy"' in report_template
