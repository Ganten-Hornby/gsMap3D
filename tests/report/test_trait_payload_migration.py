from types import SimpleNamespace

import pandas as pd

from gsMap.report import report_data
from gsMap.report.trait_metadata import trait_payload_current


def test_pcc_payload_upgrade_reuses_existing_csv_without_recomputing(tmp_path, monkeypatch):
    manager = report_data.ReportDataManager.__new__(report_data.ReportDataManager)
    manager.report_data_dir = tmp_path / "data"
    manager.js_data_dir = tmp_path / "web" / "js_data"
    manager.report_data_dir.mkdir()
    (manager.js_data_dir / "gss_stats").mkdir(parents=True)
    manager.force_re_run = False
    manager.traits = ["A B"]
    csv = manager.report_data_dir / "A B.csv"
    pd.DataFrame({"gene": ["NA"], "trait": ["A B"], "PCC": [0.42]}).to_csv(csv, index=False)
    (manager.report_data_dir / "gene_list.csv").write_text("gene\nNA\n")
    manager.report_config = SimpleNamespace(get_gene_diagnostic_info_save_path=lambda _: csv)
    js = manager.js_data_dir / "gss_stats" / "gene_trait_correlation_A B.js"
    js.write_text("window.GSMAP_GENE_TRAIT_CORRELATION_A_B=[];")

    def unexpected(*args):
        raise AssertionError("Payload upgrade must not recompute PCC or read markers")

    monkeypatch.setattr(report_data, "_load_gss_pcc_sample", unexpected)
    before = csv.read_bytes()
    manager.prepare_gss_stats()
    assert csv.read_bytes() == before
    assert trait_payload_current(js, "GSMAP_GENE_TRAIT_CORRELATION")
    assert '"gene":"NA"' in js.read_text()
