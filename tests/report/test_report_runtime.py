"""Run report JavaScript, rather than only checking template substrings."""

import json
import shutil
import subprocess

import pytest

from tests.paths import REPO_ROOT

TEMPLATE = REPO_ROOT / "src/gsMap/report/static/template.html"


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
                div.data = traces;
                div.renderCount = (div.renderCount || 0) + 1;
                return Promise.resolve();
            },
            purge(div) {div.purged = true; delete div.data;}
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


def test_spatial_maps_start_in_compare_without_2d_signal_fading(run_js):
    result = run_js("""
        console.log(JSON.stringify({
            mode2d: app.spatial2dViewMode, mode3d: app.spatial3dViewMode,
            emphasis2d: app.spatial2dAssociationOpacityStrength,
            emphasis3d: app.spatial3dAssociationOpacityStrength,
            opacity: app.associationOpacityArray(
                new Float32Array([0, 1, 5, 10]), 0, 10, 0.9,
                app.spatial2dAssociationOpacityStrength, 0),
            camera: app.spatial3dLayout().scene.camera
        }));
    """)
    assert result["mode2d"] == result["mode3d"] == "compare"
    assert result["emphasis2d"] == 0
    assert result["emphasis3d"] == 0.65
    assert result["opacity"] == 0.9
    assert result["camera"]["center"] == {"x": 0, "y": 0, "z": 0}
    assert result["camera"]["eye"] == pytest.approx({"x": 1.8, "y": 1.8, "z": 1.8})


def test_spatial2d_markers_follow_actual_viewport_zoom_and_manual_scale(run_js):
    result = run_js("""
        const data = {point_size: 6, point_geometry: {diameter: 1.4}, coordinate_ranges: [100, 200]};
        const target = {_fullLayout: {
            xaxis: {range: [0, 100], _length: 400}, yaxis: {range: [200, 0], _length: 800}}};
        const initial = app.resolveSpatial2dPointSize(data, target);
        target._fullLayout.xaxis.range = [0, 50]; target._fullLayout.yaxis.range = [100, 0];
        const zoomed = app.resolveSpatial2dPointSize(data, target);
        app.spatial2dPointScale = 2;
        const manual = app.resolveSpatial2dPointSize(data, target);
        target._fullLayout.xaxis._length /= 2; target._fullLayout.yaxis._length /= 2;
        const smaller = app.resolveSpatial2dPointSize(data, target);
        console.log(JSON.stringify({initial, zoomed, manual, smaller}));
    """)
    assert result == pytest.approx(
        {"initial": 5.6, "zoomed": 11.2, "manual": 22.4, "smaller": 11.2}
    )


def test_spatial2d_point_resize_is_bounded_and_only_changes_marker_size(run_js):
    result = run_js("""
        let callback, binds = 0; const changes = [];
        const data = {point_size: 6, point_geometry: {diameter: 20}};
        app.currentSampleData = data;
        const p = {on(name, cb) {binds++; callback = cb;},
            _fullLayout: {xaxis: {range:[0,1], _length:100}, yaxis: {range:[0,1], _length:100}},
            data: [{marker: {size:6, color:[1,2], cmin:0, cmax:10, opacity:0.9}}]};
        Plotly.restyle = (p, change) => {changes.push(change); p.data[0].marker.size = change['marker.size'];};
        app.bindSpatial2dPointSizing(p, data, 'trait');
        app.bindSpatial2dPointSizing(p, data, 'trait');
        callback(); callback();
        app.currentSampleData = {};
        callback();
        console.log(JSON.stringify({binds, changes, marker:p.data[0].marker, label:app.spatial2dPointSizePixels}));
    """)
    assert result["binds"] == 1
    assert result["changes"] == [{"marker.size": 96}]
    assert result["marker"] == {"size": 96, "color": [1, 2], "cmin": 0, "cmax": 10, "opacity": 0.9}
    assert result["label"] == 96


def test_spatial2d_point_sizing_before_first_layout_and_without_geometry(run_js):
    result = run_js("""
        const p = {clientWidth: 474, clientHeight: 820};
        const first = app.resolveSpatial2dPointSize({point_geometry: {diameter:1.4}, point_size:6,
            coordinate_ranges:[100,200]}, p);
        const old = app.resolveSpatial2dPointSize({point_size:6}, p);
        console.log(JSON.stringify({first, old}));
    """)
    assert result == pytest.approx({"first": 5.04, "old": 6})


def test_spatial2d_autorange_refinement_is_bounded(run_js):
    result = run_js("""
        let calls = 0;
        const data = {point_size:6, point_geometry:{diameter:2}};
        app.currentSampleData = data;
        const p = {data:[{marker:{size:6}}], _fullLayout:{
            xaxis:{range:[0,100],_length:400},yaxis:{range:[0,100],_length:400}}};
        app.bindSpatial2dPointSizing(p, data, 'trait');
        Plotly.restyle = (plot, change) => {
            calls++; plot.data[0].marker.size=change['marker.size'];
            plot._fullLayout.xaxis._length *= 0.8;
            plot._fullLayout.yaxis._length *= 0.8;
        };
        app.updateSpatial2dPointSize(p);
        await new Promise(resolve => setTimeout(resolve, 0));
        console.log(JSON.stringify({calls}));
    """)
    assert result["calls"] == 3


def test_umap_resize_only_when_container_dimensions_differ(run_js):
    result = run_js("""
        let calls = 0;
        Plotly.Plots = {resize(p) {
            calls++;
            p._fullLayout.width = p.clientWidth;
            p._fullLayout.height = p.clientHeight;
        }};
        const target = {data: [{}], clientWidth: 535, clientHeight: 480,
            _fullLayout: {width: 700, height: 480}};
        await app.fitUmapToContainer(target);
        await app.fitUmapToContainer(target);
        await app.fitUmapToContainer({...target, clientWidth: 0});
        console.log(JSON.stringify({calls, layout: target._fullLayout,
            autosize: app.umapLayout('test').autosize}));
    """)
    assert result == {"calls": 1, "layout": {"width": 535, "height": 480}, "autosize": True}


def test_compare_camera_preserves_rotation_and_existing_zoom_out(run_js):
    result = run_js("""
        app.spatial3dViewMode = 'compare';
        const original = {eye: {x: -1, y: 2, z: 1}, center: {x: 3, y: 0, z: 0},
            up: {x: 0, y: 1, z: 0}};
        const fitted = app.fittedSpatial3dCamera(original);
        const far = app.fittedSpatial3dCamera({eye: {x: 4, y: 4, z: 4}});
        console.log(JSON.stringify({original, fitted, far, reset: app.fittedSpatial3dCamera()}));
    """)
    eye = result["fitted"]["eye"]
    assert eye["y"] == pytest.approx(-2 * eye["x"])
    assert sum(value * value for value in eye.values()) == pytest.approx(3 * 1.8**2)
    assert result["fitted"]["center"] == {"x": 0, "y": 0, "z": 0}
    assert result["fitted"]["up"] == {"x": 0, "y": 1, "z": 0}
    assert result["original"]["center"]["x"] == 3
    assert result["far"]["eye"] == {"x": 4, "y": 4, "z": 4}
    assert result["reset"]["eye"]["x"] == pytest.approx(1.8)


def test_spatial3d_redraw_keeps_camera_and_does_not_mutate_plot_state(run_js):
    result = run_js("""
        const camera = {eye: {x: 1.8, y: -2, z: 1.2}, center: {x: 0.1, y: 0, z: 0},
            up: {x: 0, y: 0, z: 1}, projection: {type: 'perspective'}};
        const target = {_fullLayout: {scene: {camera}}};
        const layout = app.spatial3dLayout(target);
        const same = JSON.stringify(layout.scene.camera) === JSON.stringify(camera);
        layout.scene.camera.eye.x = 99;
        console.log(JSON.stringify({same, originalEyeX: camera.eye.x}));
    """)
    assert result == {"same": True, "originalEyeX": 1.8}


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


def test_trait_switch_reuses_heatmap_but_changed_view_data_or_missing_plot_redraw(run_js):
    result = run_js("""
        let rows = [{annotation: '01', trait: 'IQ', mlog10_p_cauchy: 2}];
        app.selectedAnnotationCategory = 'domain';
        app.cauchyIndex = {table: {all: {IQ: {domain: {table: true}},
            AF: {domain: {table: true}}}}};
        app.loadCauchyShard = async entry => entry?.table ? [{trait: app.selectedTrait}] : rows;
        app.selectedTrait = 'IQ';
        await app.loadActiveCauchyData();
        app.selectedTrait = 'AF';
        await app.loadActiveCauchyData();
        const same = {renders: plot.renderCount, table: app.cauchyData, matches: app.cauchyHeatmapSelectionMatches};
        app.cauchyViewMode = 'sample';
        const beforeViewLoad = app.cauchyHeatmapSelectionMatches;
        await app.loadActiveCauchyData();
        rows = [...rows, {annotation: '02', trait: 'IQ', mlog10_p_cauchy: 3}];
        await app.loadActiveCauchyData();
        Plotly.purge(plot);
        await app.loadActiveCauchyData();
        console.log(JSON.stringify({same, beforeViewLoad, renders: plot.renderCount}));
    """)
    assert result == {
        "same": {"renders": 1, "table": [{"trait": "AF"}], "matches": True},
        "beforeViewLoad": False,
        "renders": 4,
    }


def test_same_trait_requested_again_cannot_be_overwritten_by_older_request(run_js):
    result = run_js("""
        const pending = [];
        app.cauchyIndex = {table: {all: {A: {domain: {table: true}}, B: {domain: {table: true}}}}};
        app.selectedAnnotationCategory = 'domain';
        app.loadCauchyShard = entry => entry?.table
            ? new Promise(resolve => pending.push(resolve))
            : Promise.resolve([{annotation: '01', trait: 'A', mlog10_p_cauchy: 2}]);
        app.selectedTrait = 'A';
        const first = app.loadActiveCauchyData();
        app.selectedTrait = 'B';
        const second = app.loadActiveCauchyData();
        app.selectedTrait = 'A';
        const last = app.loadActiveCauchyData();
        pending[2]([{trait: 'A', version: 'latest'}]);
        await last;
        pending[0]([{trait: 'A', version: 'stale'}]);
        pending[1]([{trait: 'B', version: 'stale'}]);
        await Promise.all([first, second]);
        console.log(JSON.stringify({rows: app.cauchyData, renders: plot.renderCount, loading: app.loadingCauchyData}));
    """)
    assert result == {
        "rows": [{"trait": "A", "version": "latest"}],
        "renders": 1,
        "loading": False,
    }


@pytest.mark.parametrize("mode", ["trait", "annotation", "compare"])
def test_trait_only_render_leaves_annotation_panels_unchanged(run_js, mode):
    result = run_js(
        """
        app.spatial2dViewMode = app.spatial3dViewMode = """
        + json.dumps(mode)
        + """;
        app.plotlyReady = true;
        app.currentSampleData = {};
        app.spatial3dBase = {};
        const calls = [];
        app.renderSpatial2DPlotSingle = (div, type) => calls.push('2d-' + type);
        app.renderSpatial3dTrait = () => calls.push('3d-trait');
        app.renderSpatial3dAnnotation = () => calls.push('3d-annotation');
        app.renderSpatial2DPlot({traitOnly: true});
        await app.renderSpatial3dViews({traitOnly: true});
        const traitOnly = [...calls];
        calls.length = 0;
        app.renderSpatial2DPlot();
        await app.renderSpatial3dViews();
        console.log(JSON.stringify({traitOnly, all: calls}));
    """
    )
    assert result["traitOnly"] == ([] if mode == "annotation" else ["2d-trait", "3d-trait"])
    expected = [
        f"{dimension}-{kind}"
        for dimension in ("2d", "3d")
        for kind in ("trait", "annotation")
        if mode == "compare" or kind == mode
    ]
    assert result["all"] == expected
