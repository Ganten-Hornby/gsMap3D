"""Validate tutorial association/metadata identity and numerical output contracts."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from anndata.io import read_elem


def validate(project):
    with h5py.File(project / "latent_to_gene/concatenated_latent_adata.h5ad") as handle:
        obs = read_elem(handle["obs"])
    metadata = json.loads((project / "gsmap_web_report/report_meta.json").read_text())
    scores = json.loads((project / "latent_to_gene/marker_scores.meta.json").read_text())
    if not scores.get("complete") or scores["shape"][0] != len(obs) or not obs.index.is_unique:
        raise ValueError("Incomplete scores or invalid metadata universe")
    traits = []
    for trait in metadata["traits"]:
        path = project / "spatial_ldsc" / f"{project.name}_{trait}.csv.gz"
        frame = pd.read_csv(path)
        checks = dict(
            unique_spots=not frame.spot.duplicated().any(),
            identical_ids=len(frame)
            == len(obs)
            == len(pd.Index(frame.spot).intersection(obs.index)),
            finite_logp=bool(np.isfinite(frame.neg_log10_p).all()),
            valid_p=bool(frame.p.between(0, 1).all()),
        )
        if not all(checks.values()):
            raise ValueError(f"Association validation failed: {trait}: {checks}")
        traits.append(
            dict(
                trait=trait,
                rows=len(frame),
                max_logp=float(frame.neg_log10_p.max()),
                checks=checks,
            )
        )
    if metadata["is_3d"] != (metadata["dataset_type"] == "spatial3D"):
        raise ValueError("Report dataset mode mismatch")
    return dict(
        project=str(project), cells=len(obs), genes=scores["shape"][1], traits=traits, passed=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = [validate(project) for project in args.project]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for entry in result:
        print(
            f"Validated {Path(entry['project']).name}: {entry['cells']:,} cells, {len(entry['traits'])} traits"
        )
