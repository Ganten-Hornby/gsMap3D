import json
from types import SimpleNamespace

from gsMap.config.quick_mode_config import _report_visualization_config_matches
from gsMap.report.spatial_payload import trait_file_token
from gsMap.report.trait_metadata import trait_payload_current, trait_payload_js
from tests.report import test_report_runtime

run_js = test_report_runtime.run_js


def test_spatial_trait_filenames_are_not_lossy():
    ids = ["A B", "A_B", "A-B", "A.B", "智力"]
    assert len({trait_file_token(trait) for trait in ids}) == len(ids)
    assert trait_file_token("IQ") == "IQ"
    assert trait_file_token(trait_file_token("A B")) != trait_file_token("A B")


def test_label_and_id_fuzzy_search_keep_stable_selections(run_js):
    result = run_js("""
        app.traits=['IQ','SCZ']; app.reportMeta={trait_labels:{IQ:'Intelligence',SCZ:'Schizophrenia'}};
        app.traitSearchQuery='intellignce'; const byLabel=app.filteredTraits;
        app.traitSearchQuery='IQ'; const byId=app.filteredTraits;
        app.spatialComparisonQueries.traitB='schizo';
        const compare=app.spatialComparisonChoices({key:'traitB',values:app.traits});
        app.cauchyHeatmapData=[{trait:'IQ',annotation:'brain',mlog10_p_cauchy:3},
            {trait:'SCZ',annotation:'brain',mlog10_p_cauchy:5}];
        app.cauchyHeatmapTraitQuery='schizo'; const subset=app.heatmapSubsetChoices('trait');
        console.log(JSON.stringify({byLabel,byId,compare,subset,fallback:app.traitLabel('Height')}));
    """)
    assert result == dict(
        byLabel=["IQ"], byId=["IQ"], compare=["SCZ"], subset=["SCZ"], fallback="Height"
    )


def test_heatmap_labels_do_not_change_axes_or_merge_identical_labels(run_js):
    result = run_js("""
        app.reportMeta={trait_labels:{IQ:'Cognition',SCZ:'Cognition'}};
        app.cauchyHeatmapData=[{trait:'IQ',annotation:'brain',mlog10_p_cauchy:3},
            {trait:'SCZ',annotation:'brain',mlog10_p_cauchy:5}];
        app.cauchyHeatmapCluster=false; app.renderCauchyHeatmap();
        const ids=plot.traces[0].x, labels=plot.layout.xaxis.ticktext;
        app.cauchyHeatmapTranspose=true; app.renderCauchyHeatmap();
        const hover=app.heatmapHoverDetails({x:'brain',y:'IQ',customdata:3,
            data:{meta:{heatmapRole:'matrix',transpose:true,metricLabel:'P'}}});
        console.log(JSON.stringify({ids,labels,transposed:plot.layout.yaxis.ticktext,hover:hover.html}));
    """)
    assert set(result["ids"]) == {"IQ", "SCZ"}
    assert result["labels"] == result["transposed"] == ["Cognition", "Cognition"]
    assert "Cognition" in result["hover"]


def test_collision_free_payloads_and_cached_manhattan_switch(run_js, tmp_path):
    scripts = "\n".join(
        trait_payload_js(registry, key, json.dumps({"id": key}))
        for registry in ["GSMAP_MANHATTAN", "GSMAP_GENE_TRAIT_CORRELATION"]
        for key in ["A B", "A_B", "A-B", "智力"]
    )
    result = run_js(
        scripts
        + """
        app.renderManhattanPlot=()=>{};
        app.manhattanLoadedTraits={'A B':true,'A_B':true};
        app.selectedTrait='A B'; await app.loadManhattanData(); const first=app.manhattanData.id;
        app.selectedTrait='A_B'; await app.loadManhattanData();
        console.log(JSON.stringify({first,second:app.manhattanData.id,
            keys:Object.keys(window.GSMAP_GENE_TRAIT_CORRELATION)}));
    """
    )
    assert result["first"] == "A B"
    assert result["second"] == "A_B"
    assert len(result["keys"]) == 4
    path = tmp_path / "payload.js"
    path.write_text(trait_payload_js("GSMAP_MANHATTAN", "IQ", "{}"))
    assert trait_payload_current(path, "GSMAP_MANHATTAN")
    path.write_text("window.GSMAP_MANHATTAN_IQ={};")
    assert not trait_payload_current(path, "GSMAP_MANHATTAN")


def test_label_changes_invalidate_only_report_completion(tmp_path):
    config = SimpleNamespace(
        web_report_dir=tmp_path,
        trait_name_list=["IQ"],
        trait_labels={"IQ": "Intelligence"},
        downsampling_n_spots_umap=100,
        downsampling_n_spots_2d=100,
        downsampling_n_spots_3d=100,
        top_corr_genes=10,
    )
    meta = {
        k: getattr(config, k)
        for k in [
            "trait_labels",
            "downsampling_n_spots_umap",
            "downsampling_n_spots_2d",
            "downsampling_n_spots_3d",
            "top_corr_genes",
        ]
    }
    meta["trait_payload_version"] = 2
    (tmp_path / "report_meta.json").write_text(json.dumps(meta))
    assert _report_visualization_config_matches(config)
    config.trait_labels = {"IQ": "Intelligence — cognitive performance"}
    assert not _report_visualization_config_matches(config)
