import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from gsMap.report.report_data import _export_report_meta_js, _prepare_3d_visualization


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate only the missing Spatial 3D HTML assets for a completed report."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-observations", required=True, type=int)
    return parser.parse_args()


def _write_report_metadata(web_report_dir: Path, report_meta: dict) -> None:
    metadata_path = web_report_dir / "report_meta.json"
    partial_path = metadata_path.with_suffix(".json.partial")
    partial_path.write_text(json.dumps(report_meta), encoding="utf-8")
    partial_path.replace(metadata_path)
    _export_report_meta_js(web_report_dir, web_report_dir / "js_data", report_meta)


def main() -> None:
    args = _parse_args()
    report_data_dir = args.run_dir / "report_data"
    web_report_dir = args.run_dir / "gsmap_web_report"
    metadata_path = report_data_dir / "spot_metadata.csv"
    report_meta_path = web_report_dir / "report_meta.json"

    report_meta = json.loads(report_meta_path.read_text(encoding="utf-8"))
    traits = list(report_meta["traits"])
    annotations = list(report_meta["annotations"])
    metadata = pd.read_csv(metadata_path)
    if len(metadata) != args.expected_observations:
        raise RuntimeError(
            f"Expected {args.expected_observations} observations, found {len(metadata)}"
        )

    config = SimpleNamespace(
        annotation_list=annotations,
        downsampling_n_spots_3d=args.expected_observations,
    )
    widget_path = _prepare_3d_visualization(
        config,
        metadata,
        traits,
        report_data_dir,
        web_report_dir,
    )
    if widget_path is None:
        raise RuntimeError("Spatial 3D export returned no widget path")

    audit_path = web_report_dir / "spatial_3d" / "spatial_3d_export_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("input_observations") != args.expected_observations:
        raise RuntimeError(f"Unexpected 3D input count: {audit}")
    if audit.get("exported_observations") != args.expected_observations:
        raise RuntimeError(f"Spatial 3D export was downsampled: {audit}")
    for asset, count in audit.get("assets", {}).items():
        asset_path = web_report_dir / "spatial_3d" / asset
        if count != args.expected_observations or not asset_path.is_file():
            raise RuntimeError(f"Invalid Spatial 3D asset {asset}: {count}")
        if asset_path.stat().st_size == 0:
            raise RuntimeError(f"Empty Spatial 3D asset: {asset_path}")

    report_meta["has_3d_widget"] = True
    report_meta["spatial_3d_widget_path"] = widget_path
    _write_report_metadata(web_report_dir, report_meta)
    print(f"Recovered Spatial 3D report assets: {widget_path}")


if __name__ == "__main__":
    main()
