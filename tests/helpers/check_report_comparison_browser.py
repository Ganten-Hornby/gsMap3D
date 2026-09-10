"""Exercise comparison views on a real report, without editing its payloads.

Run with the docs/Playwright environment. A separate review copy is created using
hardlinks for immutable assets, and its HTML is replaced atomically.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from jinja2 import Template
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    report = output / "gsmap_web_report"
    if source == report or source in output.parents:
        parser.error("Use a separate output directory outside the original report.")
    output.mkdir(parents=True, exist_ok=True)
    if not report.exists():
        shutil.copytree(source, report, copy_function=os.link)
    template = Path(__file__).resolve().parents[2] / "src/gsMap/report/static/template.html"
    temporary = report / ".comparison-index.tmp"
    temporary.write_text(
        Template(template.read_text(), autoescape=True).render(
            title="gsMap Report · comparison review",
            project_name="Comparison review",
            generated_at="Comparison feature validation",
            gsmap_version="0.1.3",
        )
    )
    temporary.replace(report / "index.html")
    errors, failed = [], []
    result = {"passed": False, "report": str(report), "checks": []}

    def checked(name, details=None):
        result["checks"].append({"name": name, "details": details})
        print(json.dumps(result["checks"][-1]), flush=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1100}, accept_downloads=True)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("requestfailed", lambda request: failed.append(request.url))
        page.set_default_timeout(120000)
        try:
            page.goto((report / "index.html").as_uri(), wait_until="networkidle", timeout=120000)
            page.wait_for_function(
                "window.Alpine && Alpine.$data(document.body).dataLoaded", timeout=120000
            )
            page.locator("#sec-gsmap").scroll_into_view_if_needed()
            page.wait_for_function(
                "document.getElementById('spatial2dPlotAnnotation')?.data?.length > 0",
                timeout=120000,
            )
            page.evaluate("document.documentElement.style.scrollBehavior='auto'")
            original = page.evaluate("""() => { const a=Alpine.$data(document.body);
                return [a.selectedSample,a.selectedTrait]; }""")
            page.get_by_label("Comparison type", exact=True).select_option("traits")
            page.wait_for_function("!Alpine.$data(document.body).spatialComparisonBusy")
            names = page.evaluate("""() => {const a=Alpine.$data(document.body);
                return [a.traits.find(t=>t==='IQ'||t.startsWith('IQ_'))||a.traits[0],
                    a.traits.find(t=>t.includes('SCZ')||t==='AF')||a.traits[1]];}""")
            for label, value in zip(["Trait A", "Trait B"], names, strict=True):
                search = page.get_by_role("combobox", name=label, exact=True)
                search.fill(value)
                search.press("Enter")
                page.wait_for_function("!Alpine.$data(document.body).spatialComparisonBusy")
            search = page.get_by_role("combobox", name="Trait B", exact=True)
            search.focus()
            assert search.input_value() == ""
            search.fill("no-such-GWAS-trait-xyz")
            page.locator("#comparison-list-traitB").get_by_text(
                "No matches. Try a shorter name or abbreviation.", exact=True
            ).wait_for(state="visible")
            search.press("Escape")
            assert search.input_value() == names[1]
            checked("sidebar-style fuzzy combobox, Enter selection and Escape restoration")
            panel_state = page.evaluate("""() => {
                const a=Alpine.$data(document.body),p=document.getElementById('spatial2dPlotTrait'),q=document.getElementById('spatial2dPlotAnnotation');
                return {error:a.spatial2dError, selected:[a.selectedSample,a.selectedTrait],
                    count:p.data[0].x.length, identical:Array.from(p.data[0].x).every((x,i)=>x===q.data[0].x[i]),
                    limits:[p.data[0].marker.cmax,q.data[0].marker.cmax],
                    defaults:a.spatialComparisonPanels().map(p=>a.sampleDataCache[p.sample].traits[p.trait].cmax),
                    geometryShared:a.sampleDataCache[a.spatialComparisonSelection.sampleA].sx===a.currentSampleData.sx};
            }""")
            assert not panel_state["error"], panel_state
            assert panel_state["identical"], panel_state
            assert panel_state["limits"] == panel_state["defaults"], panel_state
            assert panel_state["selected"] == original
            checked("same sample / two traits", panel_state)
            changed = page.evaluate("""async () => {
                const a=Alpine.$data(document.body),ids=['spatial2dPlotTrait','spatial2dPlotAnnotation'];
                const third=a.traits.find(t=>!a.spatialComparisonPanels().some(p=>p.trait===t));
                const previous=a.spatialComparisonSelection.traitB;
                const counts=Object.fromEntries(ids.map(id=>[id,0]));
                const react=Plotly.react;
                Plotly.react=function(plot,...args){if(plot.id in counts) counts[plot.id]++;return react.call(this,plot,...args);};
                try {
                    const trace=document.getElementById(ids[0]).data[0];
                    await a.setSpatialComparisonChoice('traitB',third);
                    const kept=trace===document.getElementById(ids[0]).data[0];
                    const hover=a._pointHoverContexts[trace.meta.hoverContextKey].comparisonTraits.map(p=>p.trait);
                    const changed={...counts};
                    counts[ids[0]]=counts[ids[1]]=0;
                    a.spatialComparisonColorMax=2.5; await a.renderSpatialComparison();
                    const manual={counts:{...counts},limits:ids.map(id=>document.getElementById(id).data[0].marker.cmax)};
                    a.spatialComparisonColorMax=0; await a.renderSpatialComparison();
                    await a.setSpatialComparisonChoice('traitB',previous);
                    return {third,kept,hover,changed,manual};
                } finally {Plotly.react=react;}
            }""")
            assert changed["kept"]
            assert changed["hover"][1] == changed["third"]
            assert list(changed["changed"].values()) == [0, 1], changed
            assert list(changed["manual"]["counts"].values()) == [1, 1], changed
            assert changed["manual"]["limits"] == [2.5, 2.5]
            checked(
                "trait switch redraws only changed panel; manual maximum updates both", changed
            )
            page.locator("#sec-gsmap").screenshot(path=str(output / "traits.png"))
            # Real mouse drag: the right panel must move before releasing the button.
            page.locator("#spatial2dPlotTrait").scroll_into_view_if_needed()
            rect = page.locator("#spatial2dPlotTrait .nsewdrag").bounding_box()
            x, y = rect["x"] + rect["width"] * 0.4, rect["y"] + rect["height"] * 0.5
            before = page.evaluate(
                "document.getElementById('spatial2dPlotAnnotation')._fullLayout.xaxis.range"
            )
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + 40, y + 20, steps=5)
            started = time.monotonic()
            page.wait_for_function(
                """before => {
                const p=document.getElementById('spatial2dPlotTrait'),q=document.getElementById('spatial2dPlotAnnotation');
                return p._dragging && q._fullLayout.xaxis.range[0]!==before[0] &&
                    ['xaxis','yaxis'].every(k=>p._fullLayout[k].range.every((v,i)=>Math.abs(v-q._fullLayout[k].range[i])<1e-5));
            }""",
                arg=before,
                timeout=10000,
            )
            checked(
                "live pan synchronized before mouse release",
                {"check_wait_ms": (time.monotonic() - started) * 1000},
            )
            page.mouse.up()
            # Hover must report both actual array values for this exact source row.
            hover = page.evaluate("""() => {
                const a=Alpine.$data(document.body),p=document.getElementById('spatial2dPlotTrait'),t=p.data[0];
                const l=p._fullLayout,r=p.getBoundingClientRect();
                for(let i=0;i<t.x.length;i++) {
                    const x=l.xaxis.l2p(t.x[i]),y=l.yaxis.l2p(t.y[i]);
                    if(x>20&&x<l.xaxis._length-20&&y>20&&y<l.yaxis._length-20)
                        return {x:r.left+l.xaxis._offset+x,y:r.top+l.yaxis._offset+y};
                }
            }""")
            page.mouse.move(hover["x"], hover["y"])
            tooltip = page.locator("#gsmapPointHoverTooltip")
            tooltip.wait_for(state="visible")
            assert all(name in tooltip.inner_text() for name in names), tooltip.inner_text()
            checked("paired trait hover", tooltip.inner_text())
            filtered = page.evaluate("""async () => {
                const a=Alpine.$data(document.body);
                const annotation=a.sampleDataCache[a.spatialComparisonSelection.sampleA].annotations[a.selectedAnnotationCategory];
                const category=annotation.categories[annotation.codes[0]];
                a.setAnnotationCategoryVisible(category,false);
                await a.waitForLayoutPaint(); await a.renderSpatialComparison();
                return {category,counts:['spatial2dPlotTrait','spatial2dPlotAnnotation'].map(id=>document.getElementById(id).data[0].x.length)};
            }""")
            assert filtered["counts"][0] == filtered["counts"][1]
            assert filtered["counts"][0] < panel_state["count"]
            checked("shared annotation filtering keeps identical source rows", filtered)
            # A view roundtrip must preserve real zoom ranges, category filters and scaling.
            with page.expect_download() as download:
                page.get_by_role("button", name="Save view", exact=True).click()
            view_path = output / "saved-view.json"
            download.value.save_as(view_path)
            saved = json.loads(view_path.read_text())
            assert saved["kind"] == "traits"
            assert saved["selection"]["traitB"] == names[1]
            page.get_by_label("Comparison color maximum").fill("2.5")
            page.get_by_label("Comparison color maximum").press("Tab")
            page.wait_for_function("!Alpine.$data(document.body).spatialComparisonBusy")
            page.evaluate("""async () => {const a=Alpine.$data(document.body);
                a.showAllAnnotationCategories(); await a.waitForLayoutPaint(); await a.renderSpatialComparison();}""")
            page.locator("input[x-ref=comparisonFile]").set_input_files(view_path)
            page.wait_for_function(
                "Alpine.$data(document.body).spatialComparisonNotice.startsWith('View restored')"
            )
            page.wait_for_function("!Alpine.$data(document.body).spatialComparisonBusy")
            restored = page.evaluate("Alpine.$data(document.body).spatialComparisonBookmark()")
            for key in [
                "selection",
                "linked",
                "colorMax",
                "pointScale",
                "annotation",
                "hidden",
                "ranges",
            ]:
                assert restored[key] == saved[key], (key, saved[key], restored[key])
            checked("offline JSON download/import roundtrip")
            with page.expect_download(timeout=120000) as download:
                page.get_by_role("button", name="Export comparison PNG", exact=True).click()
            download.value.save_as(output / "comparison.png")
            assert (output / "comparison.png").stat().st_size > 10000
            checked("offline composite PNG export")
            page.get_by_label("Comparison type", exact=True).select_option("samples")
            page.wait_for_function("!Alpine.$data(document.body).spatialComparisonBusy")
            sample_state = page.evaluate("""() => {
                const a=Alpine.$data(document.body);
                return {panels:a.spatialComparisonPanels(), linked:a.spatialComparisonLinked,error:a.spatial2dError,
                    limits:['spatial2dPlotTrait','spatial2dPlotAnnotation'].map(id=>document.getElementById(id).data[0].marker.cmax)};
            }""")
            assert not sample_state["linked"]
            assert not sample_state["error"]
            assert sample_state["panels"][0]["trait"] == sample_state["panels"][1]["trait"]
            assert sample_state["panels"][0]["sample"] != sample_state["panels"][1]["sample"]
            assert sample_state["limits"][0] == sample_state["limits"][1]
            checked("two independent samples / same trait", sample_state)
            page.locator("#sec-gsmap").screenshot(path=str(output / "samples.png"))
            unlinked = page.evaluate("""async () => {
                const a=Alpine.$data(document.body),p=document.getElementById('spatial2dPlotTrait'),q=document.getElementById('spatial2dPlotAnnotation');
                const before=q._fullLayout.xaxis.range.slice();
                await Plotly.relayout(p,{'xaxis.range':p._fullLayout.xaxis.range.map(x=>x+5)});
                await a.waitForLayoutPaint();
                return before.every((x,i)=>x===q._fullLayout.xaxis.range[i]);
            }""")
            assert unlinked
            linked = page.evaluate("""async () => {
                const a=Alpine.$data(document.body); a.spatialComparisonLinked=true;
                await a.alignSpatialComparison(); await a.resetSpatial2dView(); await a.waitForLayoutPaint();
                const p=document.getElementById('spatial2dPlotTrait'),q=document.getElementById('spatial2dPlotAnnotation');
                return ['xaxis','yaxis'].every(k=>p._fullLayout[k].range.every((x,i)=>Math.abs(x-q._fullLayout[k].range[i])<1e-5));
            }""")
            assert linked
            checked("sample views independent by default; opt-in link and reset agree")
            annotations = page.evaluate("""async () => {
                const a=Alpine.$data(document.body);
                a.spatial2dAssociationMinLogP=100; a.showAllAnnotationCategories();
                const loader=a.loadSampleTraitData; let calls=0;
                a.loadSampleTraitData=async (...args)=>{calls++;return loader.apply(a,args);};
                try {await a.setSpatial2dCompareKind('sample_annotations');}
                finally {a.loadSampleTraitData=loader;}
                return {calls,error:a.spatial2dError,linked:a.spatialComparisonLinked,
                    panels:a.spatialComparisonPanels().map((panel,i)=>{
                        const plot=document.getElementById(i?'spatial2dPlotAnnotation':'spatial2dPlotTrait'),t=plot.data[0];
                        const context=a._pointHoverContexts[t.meta.hoverContextKey];
                        return {count:t.x.length,total:a.sampleDataCache[panel.sample].sx.length,
                            colors:context.annotation.categories.map((c,j)=>[String(c),context.annotation.colors[j]]),
                            prefix:context.hoverPrefix,showscale:t.marker.showscale};
                    })};
            }""")
            assert not annotations["calls"], annotations
            assert not annotations["error"]
            assert not annotations["linked"]
            for panel in annotations["panels"]:
                assert panel["count"] == panel["total"]
                assert not panel["showscale"]
            a_colors, b_colors = [dict(p["colors"]) for p in annotations["panels"]]
            assert all(a_colors[c] == b_colors[c] for c in a_colors.keys() & b_colors.keys())
            assert page.get_by_label("Comparison color maximum").is_hidden()
            assert page.get_by_label("Minimum 2D association minus log10 P").is_hidden()
            assert page.get_by_role("combobox", name="Shared annotation", exact=True).is_visible()
            checked("two samples / shared annotation colors; no GWAS load or cutoff", annotations)
            page.locator("#sec-gsmap").screenshot(path=str(output / "sample-annotations.png"))
            with page.expect_download() as download:
                page.get_by_role("button", name="Save view", exact=True).click()
            annotation_view = output / "annotation-view.json"
            download.value.save_as(annotation_view)
            assert json.loads(annotation_view.read_text())["kind"] == "sample_annotations"
            with page.expect_download(timeout=120000) as download:
                page.get_by_role("button", name="Export comparison PNG", exact=True).click()
            download.value.save_as(output / "annotation-comparison.png")
            assert (output / "annotation-comparison.png").stat().st_size > 10000
            page.locator("input[x-ref=comparisonFile]").set_input_files(annotation_view)
            page.wait_for_function(
                "Alpine.$data(document.body).spatialComparisonNotice.startsWith('View restored')"
            )
            checked("annotation comparison JSON roundtrip and PNG with shared legend")
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(1000)
            page.locator("#spatial-comparison-controls").scroll_into_view_if_needed()
            page.screenshot(path=str(output / "mobile.png"))
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
            checked("mobile layout without horizontal overflow")
            page.set_viewport_size({"width": 1500, "height": 1100})
            page.evaluate("Alpine.$data(document.body).spatial2dAssociationMinLogP=0")
            page.get_by_label("Comparison type", exact=True).select_option("annotation")
            page.wait_for_function("""() => {const p=document.getElementById('spatial2dPlotAnnotation');
                return p.data[0].meta.hoverContextKey.includes('annotation');}""")
            checked("original trait / annotation comparison retained")
            if page.evaluate("Alpine.$data(document.body).has3dWidget"):
                page.wait_for_function(
                    """['spatial3dPlotTrait','spatial3dPlotAnnotation'].every(id =>
                    document.getElementById(id)?.data?.some(trace=>trace.x?.length>0))""",
                    timeout=120000,
                )
                page.locator("#spatial3dPlotTrait").scroll_into_view_if_needed()
                page.locator("#sec-spatial3d").screenshot(path=str(output / "original-3d.png"))
                checked("original 3D views remain populated after comparison and restore")
            assert not errors, errors
            assert not failed, failed
            result["passed"] = True
        finally:
            result.update(errors=errors, failed_requests=failed)
            (output / "browser-validation.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            browser.close()


if __name__ == "__main__":
    main()
