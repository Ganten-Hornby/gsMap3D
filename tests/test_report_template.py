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
    assert report_template.count('class="plot-grid"') == 4
    assert "spatial2dViewMode === 'compare' ? 'plot-grid'" in report_template


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


def test_spatial3d_compare_waits_for_layout_then_resizes_and_recenters(
    report_template: str,
) -> None:
    assert "await this.waitForLayoutPaint()" in report_template
    assert "await this.resizeSpatial3dViews({ center: mode === 'compare' })" in report_template
    assert "Plotly.Plots.resize(plot)" in report_template
    assert "'scene.camera.center': origin" in report_template


def test_large_secondary_datasets_are_loaded_near_their_sections(
    report_template: str,
) -> None:
    assert '<script src="js_data/cauchy_results.js"></script>' not in report_template
    assert '<script src="js_data/umap_data.js"></script>' not in report_template
    assert "setupDeferredPlotLoading()" in report_template
    assert "loadUmapData()" in report_template
    assert "loadCauchyData()" in report_template
    assert 'loading="lazy"' in report_template


def test_trait_selector_supports_fuzzy_keyboard_search(report_template: str) -> None:
    assert 'role="combobox"' in report_template
    assert 'id="trait-listbox"' in report_template
    assert "scoreTraitMatch(label, query)" in report_template
    assert "moveTraitSearch(1)" in report_template
    assert "selectActiveTrait()" in report_template


def test_spatial2d_has_scalable_controls_and_synchronized_comparison(
    report_template: str,
) -> None:
    assert "spatial2dViewMode: 'compare'" in report_template
    assert "setSpatial2dViewMode('annotation')" in report_template
    assert 'aria-label="2D point-size multiplier"' in report_template
    assert 'max="12" step="0.25"' in report_template
    assert 'aria-label="2D point opacity"' not in report_template
    assert "spatial2dOpacity" not in report_template
    assert "bindSpatial2dViewportSync" in report_template
    assert "Retained high-signal spots" in report_template
    assert "Color scale is shared across samples" in report_template


def test_spatial_views_can_toggle_retained_hotspot_emphasis(report_template: str) -> None:
    assert "spatial3dHighlightHotspots: true" in report_template
    assert "spatial2dHighlightHotspots" not in report_template
    assert report_template.count("Top 500 highlighted") == 1
    assert report_template.count("Top 500 normal size") == 1
    assert "const highlight = this.spatial3dHighlightHotspots" in report_template
    assert "size: Math.max(2.5, pointSize * 1.5)" in report_template


def test_spatial_views_offer_scalable_hover_details(report_template: str) -> None:
    assert report_template.count("<b>Retained top signal</b>") == 2
    assert "−log10(P): %{marker.color:.2f}" in report_template
    assert "annotationHoverLabels" in report_template
    assert "base.n_points <= 200000" in report_template
    assert "buildSpatial2dAnnotationHoverText(data)" in report_template
    assert "compactTraitPayload.extra.annotation_codes" in report_template
    assert "Object.entries(data.annotations || {})" in report_template
    assert "<b>${anno}</b> code: %{marker.color:.0f}" not in report_template


def test_umap_uses_readable_geometry_hover_and_canonical_annotation_colors(
    report_template: str,
) -> None:
    assert 'class="umap-frame"' in report_template
    assert "resolveUmapPointSize" in report_template
    assert "scaleanchor: 'y'" in report_template
    assert "strongest trait signals are drawn on top" in report_template
    assert "currentAnnotationLegend" in report_template
    assert "this.getAnnotationColorMap(anno)" in report_template
    hover_builder = report_template[
        report_template.index("buildUmapHoverText(includeTrait = false)") : report_template.index(
            "// Render only trait-colored UMAP plots"
        )
    ]
    assert "<b>Spot:</b>" not in hover_builder
    assert "<b>Sample:</b>" not in hover_builder
    assert "bindUmapHoverTooltip" in report_template
    assert "pointer.clientX + gap" in report_template
    assert "customdata: rendering.hoverColors" in report_template
    assert "customdata: indices.map(() => color)" in report_template


def test_shared_annotation_selection_filters_all_point_maps(report_template: str) -> None:
    assert "visibleUmapPointIndices()" in report_template
    assert "const annotation = this.spatial3dAnnotationCache?.[annotationName]" in report_template
    assert "feature.extra.annotation_codes?.[annotationName]" in report_template
    assert "compactTraitPayload.extra.annotation_codes?.[annotationName]" in report_template
    assert "this.renderSpatial2DPlotSingle('spatial2dPlotTrait', 'trait')" in report_template
    assert (
        "if (this.spatial3dViewMode !== 'annotation') this.renderSpatial3dTrait()"
        in report_template
    )


def test_enrichment_uses_shared_annotation_color_reference(report_template: str) -> None:
    assert '<script src="js_data/annotation_schemas.js"></script>' in report_template
    assert "window.GSMAP_ANNOTATION_SCHEMAS" in report_template
    assert "getAnnotationColor(row.annotation, row.annotation_name)" in report_template
    assert "annotationColorTrace" in report_template
    assert "Same color as spatial and UMAP views" in report_template
    assert "toggleCauchyAnnotationFocus(row.annotation)" in report_template
    assert "clearCauchyAnnotationFocus()" in report_template

    table_filter = report_template[
        report_template.index("get filteredCauchyTable()") : report_template.index(
            "// Wait for Plotly to load"
        )
    ]
    heatmap = report_template[
        report_template.index("                renderCauchyHeatmap() {") : report_template.index(
            "// Simple Hierarchical Clustering"
        )
    ]
    assert "isAnnotationCategoryVisible" not in table_filter
    assert "isAnnotationCategoryVisible" not in heatmap


def test_annotation_first_layout_and_shared_interactive_legend(report_template: str) -> None:
    section_order = [
        report_template.index("{ key: 'show3d', id: 'sec-spatial3d'"),
        report_template.index("{ key: 'show2dDistribution', id: 'sec-gsmap'"),
        report_template.index("{ key: 'showUMAP', id: 'sec-umap'"),
        report_template.index("{ key: 'showEnrichment', id: 'sec-cauchy'"),
        report_template.index("{ key: 'showGSS', id: 'sec-genes'"),
    ]
    assert section_order == sorted(section_order)
    assert 'style="order: 1"' in report_template
    assert 'style="order: 2"' in report_template
    assert 'style="order: 3"' in report_template
    assert 'style="order: 4"' in report_template
    assert "Click to hide; double-click to isolate in point maps." in report_template
    assert "scheduleAnnotationLegendToggle(entry.category)" in report_template
    assert "isolateAnnotationCategory(entry.category)" in report_template
    assert "filterCategoricalPlotPoints" in report_template
    assert "visible: this.isAnnotationCategoryVisible(val, annoName)" in report_template


def test_dataset_counts_are_only_in_top_overview(report_template: str) -> None:
    assert report_template.count('x-text="traits.length.toLocaleString()"') == 1
    assert report_template.count('x-text="samples.length.toLocaleString()"') == 1
    assert report_template.count('x-text="annotations.length.toLocaleString()"') == 1
    assert ">Dataset</h3>" not in report_template
