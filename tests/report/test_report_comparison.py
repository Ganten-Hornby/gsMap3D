"""Independent comparison selections, data alignment, and offline view state."""

import pytest

from tests.report import test_report_runtime

run_js = test_report_runtime.run_js


SETUP = """
    app.samples = ['slice1', 'slice2']; app.traits = ['IQ', 'SCZ', 'Height'];
    app.selectedSample = 'slice1'; app.selectedTrait = 'IQ';
    app.annotations = ['celltype']; app.selectedAnnotationCategory = 'celltype';
    app.currentSampleData = {sx: new Float32Array([1,2,3]), sy: new Float32Array([4,5,6])};
    app.plotlyReady = true;
    app.waitForLayoutPaint = async () => {};
    app.alignSpatialComparison = async () => {};
    app.loadSampleData = async () => app.currentSampleData;
    app.loadSampleTraitData = async (sample, trait) => ({
        values: new Float32Array(trait === 'IQ' ? [0, 2, 6] : [1, 8, 0]),
        cmax: trait === 'IQ' ? 6 : 8, q95: trait === 'IQ' ? 4 : 5});
    const rendered = [];
    app.renderSpatial2DPlotSingle = async (id, type, options) => {
        rendered.push({id, type, sample:options.sample, trait:options.trait,
            cmax:options.cmax, q95:options.q95, union:Array.from(options.selectionValues || [])});
    };
"""


def test_comparison_selections_and_default_color_limits_are_independent(run_js):
    result = run_js(
        SETUP
        + """
        await app.setSpatial2dCompareKind('traits');
        await app.setSpatialComparisonChoice('traitB', 'Height');
        const traitLink = app.spatialComparisonLinked;
        await app.setSpatial2dCompareKind('samples');
        console.log(JSON.stringify({rendered, traitLink, sampleLink:app.spatialComparisonLinked,
            selected:[app.selectedSample,app.selectedTrait], panels:app.spatialComparisonPanels(),
            error:app.spatial2dError}));
    """
    )
    assert result["selected"] == ["slice1", "IQ"]
    assert result["traitLink"]
    assert not result["sampleLink"]
    assert result["panels"] == [
        {"sample": "slice1", "trait": "IQ"},
        {"sample": "slice2", "trait": "IQ"},
    ]
    assert [r["trait"] for r in result["rendered"][:4]] == ["IQ", "SCZ", "IQ", "Height"]
    assert [(r["cmax"], r["q95"]) for r in result["rendered"][:4]] == [
        (6, 4),
        (8, 5),
        (6, 4),
        (8, 5),
    ]
    assert result["error"] == ""


def test_mode_transition_is_busy_before_layout_or_io(run_js):
    result = run_js(
        SETUP
        + """
        let release;
        app.waitForLayoutPaint=()=>new Promise(resolve=>{release=resolve;});
        const changing=app.setSpatial2dCompareKind('traits');
        const busy=app.spatialComparisonBusy,panels=app.spatialComparisonPanelBusy.slice();
        app.waitForLayoutPaint=async()=>{}; release(); await changing;
        console.log(JSON.stringify({busy,panels,done:app.spatialComparisonBusy}));
    """
    )
    assert result == {"busy": True, "panels": [True, True], "done": False}


def test_comparison_cutoff_uses_union_and_legacy_values_are_accepted(run_js):
    result = run_js(
        SETUP
        + """
        app.spatial2dAssociationMinLogP = 3;
        await app.setSpatial2dCompareKind('traits');
        app.loadSampleTraitData = async () => new Float32Array([1,2,3]);
        await app.renderSpatialComparison();
        console.log(JSON.stringify({rendered, error:app.spatial2dError}));
    """
    )
    assert result["rendered"][0]["union"] == result["rendered"][1]["union"] == [1, 8, 6]
    assert result["rendered"][2]["cmax"] == result["rendered"][3]["cmax"] == 3
    assert result["error"] == ""


def test_comparison_respects_zero_trait_q95(run_js):
    result = run_js(
        SETUP
        + """
        app.loadSampleTraitData=async()=>({values:new Float32Array([0,0,1]),cmax:1,q95:0});
        await app.setSpatial2dCompareKind('traits');
        console.log(JSON.stringify(rendered.map(r=>r.q95)));
    """
    )
    assert result == [0, 0]


def test_comparison_coalesces_stale_loads_including_errors(run_js):
    result = run_js(
        SETUP
        + """
        app.spatial2dCompareKind = 'traits';
        app.spatialComparisonSelection = {sampleA:'slice1',sampleB:'slice2',traitA:'IQ',traitB:'SCZ'};
        let release;
        const original = app.loadSampleTraitData;
        app.loadSampleTraitData = async (sample, trait) => {
            if (trait === 'IQ') await new Promise((resolve, reject) => {release = () => reject(new Error('stale'));});
            return original(sample, trait);
        };
        const old = app.renderSpatialComparison();
        await new Promise(resolve => setTimeout(resolve,0));
        app.spatialComparisonSelection.traitA = 'Height';
        const latest = app.renderSpatialComparison();
        release(); await Promise.all([old,latest]);
        console.log(JSON.stringify({rendered,busy:app.spatialComparisonBusy,error:app.spatial2dError}));
    """
    )
    assert [r["trait"] for r in result["rendered"]] == ["Height", "SCZ"]
    assert not result["busy"]
    assert not result["error"]


def test_comparison_union_keeps_finite_signal_when_other_trait_missing(run_js):
    result = run_js(
        SETUP
        + """
        app.spatial2dAssociationMinLogP = 3;
        app.loadSampleTraitData = async (sample, trait) => ({
            values: new Float32Array(trait === 'IQ' ? [NaN, 4, 0] : [5, NaN, 0]), cmax:5});
        await app.setSpatial2dCompareKind('traits');
        console.log(JSON.stringify(rendered));
    """
    )
    assert result[0]["union"] == result[1]["union"] == [5, 4, 0]


def test_unlinked_samples_do_not_apply_queued_viewports(run_js):
    result = run_js("""
        app.spatial2dCompareKind='samples'; app.spatialComparisonLinked=false;
        const target={data:[{}]}; let calls=0;
        Plotly.relayout=async () => {calls++;};
        await app.queueSpatial2dViewport(target, {'xaxis.range':[1,2]});
        console.log(JSON.stringify({calls, pending:target.__gsMapPendingViewport,
            applying:target.__gsMapApplyingViewport}));
    """)
    assert result == {"calls": 0, "pending": None, "applying": False}


def test_comparison_missing_values_are_error_not_zero_signal(run_js):
    result = run_js(
        SETUP
        + """
        app.loadSampleTraitData = async () => null;
        await app.setSpatial2dCompareKind('traits');
        console.log(JSON.stringify({error:app.spatial2dError,rendered,busy:app.spatialComparisonBusy}));
    """
    )
    assert "misaligned" in result["error"]
    assert result["rendered"] == []
    assert not result["busy"]


def test_comparison_fuzzy_search_filters_without_mutating_selection(run_js):
    result = run_js(
        SETUP
        + """
        app.spatialComparisonSelection.traitB='Height';
        app.spatialComparisonQueries.traitB='scz';
        console.log(JSON.stringify({matches:app.spatialComparisonChoices({key:'traitB',values:app.traits}),
            selected:app.spatialComparisonSelection.traitB}));
    """
    )
    assert result == {"matches": ["SCZ"], "selected": "Height"}


def test_comparison_combobox_keyboard_and_no_match(run_js):
    result = run_js(
        SETUP
        + """
        app.spatial2dCompareKind='traits';
        app.spatialComparisonSelection.traitB='Height';
        const field=app.spatialComparisonFields.find(f=>f.key==='traitB');
        app.openSpatialComparisonSearch('traitB');
        const opened=app.spatialComparisonQueries.traitB;
        app.spatialComparisonQueries.traitB='s z';
        await app.selectActiveSpatialComparisonChoice(field);
        const selected=app.spatialComparisonSelection.traitB;
        app.openSpatialComparisonSearch('traitB');
        app.spatialComparisonQueries.traitB='no-such-trait';
        app.moveSpatialComparisonSearch(field,1);
        await app.selectActiveSpatialComparisonChoice(field);
        app.closeSpatialComparisonSearch('traitB');
        console.log(JSON.stringify({opened,selected,after:app.spatialComparisonSelection.traitB,
            field:app.spatialComparisonSearchField,query:app.spatialComparisonQueries.traitB}));
    """
    )
    assert result == {"opened": "", "selected": "SCZ", "after": "SCZ", "field": "", "query": ""}


def test_comparison_only_redraws_changed_panel_and_refreshes_kept_hover(run_js):
    result = run_js(
        SETUP
        + """
        const plots={};
        document.getElementById=id=>plots[id] ||= {data:[]};
        const originalLoad=app.loadSampleTraitData,cache={};
        app.loadSampleTraitData=async (s,t)=>cache[s+t] ||= await originalLoad(s,t);
        app.renderSpatial2DPlotSingle=async (id,type,options)=> {
            rendered.push(id);
            plots[id].data=[{meta:{hoverContextKey:id},marker:{color:options.payload.values}}];
            app._pointHoverContexts[id]={comparisonTraits:options.comparisonTraits};
        };
        await app.setSpatial2dCompareKind('traits'); rendered.length=0;
        const kept=plots.spatial2dPlotTrait.data[0];
        await app.setSpatialComparisonChoice('traitB','Height');
        const onlyChanged=rendered.slice(),sameTrace=kept===plots.spatial2dPlotTrait.data[0];
        const hover=app._pointHoverContexts.spatial2dPlotTrait.comparisonTraits.map(x=>x.trait);
        rendered.length=0; app.spatialComparisonColorMax=10; await app.renderSpatialComparison();
        const manual=rendered.slice();
        rendered.length=0; await app.setSpatialComparisonChoice('traitB','SCZ');
        const manualKeep=rendered.slice();
        app.spatial2dAssociationMinLogP=3; await app.renderSpatialComparison();
        rendered.length=0; await app.setSpatialComparisonChoice('traitB','Height');
        console.log(JSON.stringify({onlyChanged,sameTrace,hover,manual,manualKeep,union:rendered,
            limits:Object.values(plots).map(p=>p.__gsMapComparisonRender.cmax)}));
    """
    )
    both = ["spatial2dPlotTrait", "spatial2dPlotAnnotation"]
    assert result["onlyChanged"] == result["manualKeep"] == both[1:]
    assert result["sameTrace"]
    assert result["hover"] == ["IQ", "Height"]
    assert result["manual"] == result["union"] == both
    assert result["limits"] == [10, 10]


def test_two_sample_annotations_load_no_gwas_and_expose_shared_category(run_js):
    result = run_js(
        SETUP
        + """
        app.currentSampleData.annotations={celltype:{categories:['0','1'],colors:['red','blue'],codes:new Uint8Array([0,1,0])}};
        app.loadSampleTraitData=async ()=>{throw new Error('Must not load GWAS');};
        app.spatial2dAssociationMinLogP=100;
        await app.setSpatial2dCompareKind('sample_annotations');
        console.log(JSON.stringify({rendered,error:app.spatial2dError,
            fields:app.spatialComparisonFields.map(x=>x.key),linked:app.spatialComparisonLinked}));
    """
    )
    assert not result["error"]
    assert not result["linked"]
    assert result["fields"] == ["annotation", "sampleA", "sampleB"]
    assert [p["type"] for p in result["rendered"]] == ["annotation", "annotation"]
    assert [p["sample"] for p in result["rendered"]] == ["slice1", "slice2"]


def test_annotation_comparison_ignores_cutoff_and_matches_color_by_label(run_js):
    result = run_js("""
        app.selectedAnnotationCategory='celltype'; app.selectedTrait='IQ';
        app.spatial2dCompareKind='sample_annotations'; app.spatial2dAssociationMinLogP=100;
        app.getAnnotationSchema=()=>({categories:['0','1'],colors:['red','blue']});
        const target={__gsMapComparisonRender:{key:'stale'}};
        document.getElementById=()=>target;
        app.hidePointHoverTooltip=()=>{}; app.bindSpatial2dPointSizing=()=>{};
        app.updateSpatial2dPointSize=()=>{}; app.bindSpatial2dHover=()=>{};
        app.bindSpatial2dViewportSync=()=>{};
        const annotation={categories:['1','0'],colors:['green','orange'],codes:new Uint8Array([0,1,1])};
        const data={sx:new Float32Array([1,2,3]),sy:new Float32Array([4,5,6]),
            annotations:{celltype:annotation},traits:{IQ:{values:new Float32Array([0,0,0])}}};
        await app.renderSpatial2DPlotSingle('right','annotation',{sample:'slice2',data,comparison:true});
        const trace=target.data[0],context=app._pointHoverContexts[trace.meta.hoverContextKey];
        console.log(JSON.stringify({count:trace.x.length,colors:context.annotation.colors,
            original:annotation.colors,prefix:context.hoverPrefix,scale:trace.marker.showscale,
            cached:target.__gsMapComparisonRender,margin:target.layout.margin.r}));
    """)
    assert result == {
        "count": 3,
        "colors": ["blue", "red"],
        "original": ["green", "orange"],
        "prefix": "slice2 · celltype",
        "scale": False,
        "cached": None,
        "margin": 12,
    }


BOOKMARK = (
    SETUP
    + """
    app.getAnnotationSchema = () => ({categories:['Brain','Heart'],colors:['red','blue']});
    app.spatial2dCompareKind = 'traits';
    app.spatialComparisonSelection = {sampleA:'slice1',sampleB:'slice2',traitA:'IQ',traitB:'SCZ'};
    const view = app.spatialComparisonBookmark();
    view.ranges = [[[0,10],[10,0]], [[0,10],[10,0]]];
"""
)


@pytest.mark.parametrize(
    "mutation",
    [
        "view.version=2",
        "view.report='other'",
        "view.selection.traitB='absent'",
        "view.ranges[0][0]=[0,0]",
        "view.ranges[0][1]=[0,Infinity]",
        "view.colorMax=-1",
        "view.pointScale=1000",
        "view.cutoff='10'",
        "view.hidden=['not a category']",
        "view.annotation='unknown'",
        "view.linked='false'",
    ],
)
def test_invalid_bookmark_rejected_before_mutation(run_js, mutation):
    result = run_js(
        BOOKMARK
        + mutation
        + ";"
        + """
        let rejected = false;
        try {await app.restoreSpatialComparison(view);} catch (_) {rejected=true;}
        console.log(JSON.stringify({rejected,selected:app.selectedTrait,rendered}));
    """
    )
    assert result == {"rejected": True, "selected": "IQ", "rendered": []}


def test_valid_bookmark_preserves_reversed_y_ranges(run_js):
    result = run_js(
        BOOKMARK
        + """
        console.log(JSON.stringify(app.validateSpatialComparisonBookmark(view).ranges));
    """
    )
    assert result == [[[0, 10], [10, 0]], [[0, 10], [10, 0]]]


def test_annotation_comparison_bookmark_is_supported(run_js):
    result = run_js(
        BOOKMARK
        + """
        view.kind='sample_annotations';
        console.log(JSON.stringify(app.validateSpatialComparisonBookmark(view).kind));
    """
    )
    assert result == "sample_annotations"


def test_restoring_bookmark_updates_the_unified_annotation_selection(run_js):
    result = run_js(
        BOOKMARK
        + """
        const plots = {};
        document.getElementById = id => plots[id] ||= {data:[]};
        Plotly.relayout = async () => {};
        app.cauchyHeatmapSelectedAnnotations = ['Brain'];
        app.renderSpatial3dViews = async () => {};
        app.renderAnnotationCategoryViews = () => {};
        app.onAnnotationChange = async () => {app.cauchyHeatmapSelectedAnnotations=[];};
        view.hidden=['Heart'];
        await app.restoreSpatialComparison(view);
        console.log(JSON.stringify({selected:app.selectedAnnotationCategories,
            heatmapSubset:app.cauchyHeatmapSelectedAnnotations}));
    """
    )
    assert result == {"selected": ["Brain"], "heatmapSubset": []}


def test_independent_renderer_uses_source_rows_for_paired_hover(run_js):
    result = run_js("""
        app.selectedSample='sidebar'; app.selectedTrait='sidebar';
        app.selectedAnnotationCategory='celltype';
        app.spatial2dAssociationMinLogP=3; app.spatial2dAssociationOpacityStrength=.5;
        app.selectedAnnotationCategories=['Brain'];
        const target={};
        document.getElementById = () => target;
        app.hidePointHoverTooltip=()=>{};
        app.bindSpatial2dPointSizing=()=>{}; app.updateSpatial2dPointSize=()=>{};
        app.bindSpatial2dHover=()=>{}; app.bindSpatial2dViewportSync=()=>{};
        const data={sx:new Float32Array([1,2,3,4]),sy:new Float32Array([5,6,7,8]),
            annotations:{celltype:{categories:['Brain','Heart'],colors:['red','blue'],codes:new Uint8Array([0,1,0,0])}}};
        const values=new Float32Array([0,2,6,0]);
        await app.renderSpatial2DPlotSingle('left','trait',{data,sample:'slice1',trait:'IQ',
            payload:{values},comparison:true,cmax:8,q95:5,
            selectionValues:new Float32Array([4,8,6,0]),comparisonTraits:[{trait:'IQ',values}]});
        const trace=target.data[0],context=app._pointHoverContexts[trace.meta.hoverContextKey];
        console.log(JSON.stringify({x:Array.from(trace.x),values:Array.from(trace.marker.color),
            indices:Array.from(context.indices),cmax:trace.marker.cmax,cmin:trace.marker.cmin,
            opacity:Array.from(trace.marker.opacity),prefix:context.hoverPrefix,
            trait:context.comparisonTraits[0].trait}));
    """)
    assert result["x"] == [1, 3]
    assert result["values"] == [0, 6]
    assert result["indices"] == [0, 2]
    assert result["cmax"] == 8
    assert result["cmin"] == 0
    assert all(value > 0 for value in result["opacity"])
    assert result["trait"] == "IQ"
    assert result["prefix"] == "slice1 · IQ"
