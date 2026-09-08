"""Run report JavaScript, rather than only checking template substrings."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).parents[1] / "src/gsMap/report/static/template.html"


@pytest.fixture
def run_js():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise report JavaScript")
    text = TEMPLATE.read_text()
    app_code = text[text.index("function gsMapApp()") :].split("</script>", 1)[0]
    setup = """
        const plot = {replaceChildren() {this.cleared = true;}};
        global.window = {};
        global.document = {getElementById: id => id === 'cauchyHeatmap' ? plot : null};
        global.Plotly = {
            react(div, traces, layout) {
                div.traces = traces; div.layout = layout;
                return Promise.resolve();
            },
            purge(div) {div.purged = true;}
        };
        const app = gsMapApp();
        app.getPlotlyConfig = () => ({});
        app.$nextTick = fn => fn();
    """

    def run(body):
        result = subprocess.run(
            [node, "--input-type=module"],
            input=app_code + setup + body,
            text=True,
            capture_output=True,
            check=True,
        )
        return json.loads(result.stdout)

    return run


def test_numeric_labels_are_explicit_categories_after_significance_sort(run_js):
    result = run_js("""
        app.cauchyHeatmapData = [
            {annotation: 0, trait: '01', mlog10_p_cauchy: 5},
            {annotation: '10', trait: '01', mlog10_p_cauchy: 1},
            {annotation: 2, trait: '01', mlog10_p_cauchy: 8},
        ];
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify(plot));
    """)
    assert result["traces"][0]["y"] == ["10", "0", "2"]
    assert result["layout"]["yaxis"]["categoryarray"] == ["10", "0", "2"]
    assert result["layout"]["yaxis"]["type"] == result["layout"]["xaxis"]["type"] == "category"
    assert result["traces"][1]["zmin"] == -0.5
    assert result["traces"][1]["zmax"] == 2.5


def test_missing_results_stay_blank_and_do_not_bias_normalization(run_js):
    result = run_js("""
        app.cauchyHeatmapNormalize = true;
        app.cauchyHeatmapData = [
            {annotation: 'NA', trait: 't', mlog10_p_cauchy: null},
            {annotation: '01', trait: 't', mlog10_p_cauchy: 2},
            {annotation: '02', trait: 't', mlog10_p_cauchy: 4},
        ];
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify(plot.traces[0]));
    """)
    assert result["z"] == [[None], [0], [1]]
    assert result["customdata"] == [[None], [2], [4]]
    assert result["hoverongaps"] is False
    assert result["zmin"] == 0
    assert result["zmax"] == 1
    assert "0–1" in result["colorbar"]["title"]["text"]


def test_empty_selection_clears_old_heatmap(run_js):
    result = run_js("""
        app.cauchyHeatmapData = [];
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify(plot));
    """)
    assert result["purged"]
    assert result["cleared"]


def test_discrete_scale_matches_half_integer_color_limits(run_js):
    scales = run_js("""
        console.log(JSON.stringify(Array.from({length: 64}, (_, i) =>
            app.discreteColorScale(Array.from({length: i + 1}, (_, j) => 'color' + j))
        )));
    """)
    for count, scale in enumerate(scales, start=1):
        for index in range(count):
            position = (index + 0.5) / count
            lower = next(color for bound, color in reversed(scale) if bound <= position)
            upper = next(color for bound, color in scale if bound >= position)
            assert lower == upper == f"color{index}"


def test_enrichment_load_failure_clears_stale_results_and_explains_error(run_js):
    result = run_js("""
        app.cauchyIndex = {table: {}, heatmap: {}};
        app.cauchyData = [{annotation: 'old'}];
        app.cauchyHeatmapData = [{annotation: 'old'}];
        app.loadCauchyShard = () => Promise.reject(new Error('Missing data file'));
        try {await app.loadActiveCauchyData();} catch (_) {}
        console.log(JSON.stringify({
            error: app.cauchyError, loading: app.loadingCauchyData,
            rows: app.cauchyData, heatmap: app.cauchyHeatmapData, purged: plot.purged
        }));
    """)
    assert result == {
        "error": "Missing data file",
        "loading": False,
        "rows": [],
        "heatmap": [],
        "purged": True,
    }


def test_clustering_with_sparse_trait_annotation_pairs_keeps_gaps(run_js):
    result = run_js("""
        app.cauchyHeatmapData = [
            {annotation: 'a', trait: 't1', mlog10_p_cauchy: 2},
            {annotation: 'b', trait: 't1', mlog10_p_cauchy: 4},
            {annotation: 'a', trait: 't2', mlog10_p_cauchy: 3},
        ];
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify(plot.traces[0]));
    """)
    assert sorted(value for row in result["z"] for value in row if value is not None) == [2, 3, 4]
    assert sum(value is None for row in result["z"] for value in row) == 1
