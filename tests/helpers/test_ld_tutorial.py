"""Run every displayed LD recipe and compare CLI/Python outputs on real inputs."""

import argparse
import json
import os
import sys
from pathlib import Path

from tests.helpers.audit_commands import blocks
from tests.helpers.validate_examples import execute, validation_environment


def ld_tutorial_blocks(document):
    recipes = {"cli": [], "python": []}
    for _, language, code in blocks(document):
        if language == "bash" and "gsmap ldscore-weight-matrix" in code:
            recipes["cli"].append(code)
        elif language == "python" and "LDScoreConfig(" in code:
            recipes["python"].append(code)
    if not recipes["cli"] or len(recipes["cli"]) != len(recipes["python"]):
        raise ValueError("Each LD example needs one CLI and one Python recipe")
    return recipes


def compare_outputs(output):
    import anndata as ad
    import numpy as np
    import pandas as pd

    records = []
    matrices = sorted((output / "cli/results").glob("*/*.h5ad"))
    if len(matrices) != 2:
        raise ValueError(f"Expected two CLI matrices, found {len(matrices)}")
    for path in matrices:
        relative = path.relative_to(output / "cli")
        first, second = ad.read_h5ad(path), ad.read_h5ad(output / "python" / relative)
        pd.testing.assert_frame_equal(first.obs, second.obs)
        pd.testing.assert_frame_equal(first.var, second.var)
        difference = (first.X - second.X).tocsr()
        max_difference = float(np.max(np.abs(difference.data), initial=0))
        assert max_difference <= 1e-6, (path, max_difference)
        assert first.n_obs > 0
        assert first.n_vars > 1
        assert "Unmapped" in first.var_names
        assert set(first.obs["CHR"].astype(str)) == {"22"}
        assert np.isfinite(first.X.data).all()
        assert first.X[:, first.var_names != "Unmapped"].nnz > 0
        weights = sorted((path.parent / "w_ld").glob("*.l2.ldscore.gz"))
        assert len(weights) == 1, weights
        weight = pd.read_csv(weights[0], sep="\t")
        other = output / "python" / weights[0].relative_to(output / "cli")
        pd.testing.assert_frame_equal(weight, pd.read_csv(other, sep="\t"))
        assert set(weight["SNP"]) == set(first.obs_names)
        records.append(
            {
                "matrix": str(relative),
                "shape": list(first.shape),
                "max_abs_cli_python_difference": max_difference,
                "weight_rows": len(weight),
                "weights_identical": True,
            }
        )
    (output / "validation.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    recipes = ld_tutorial_blocks(repo / "docs/source/ldscore_weight_matrix.md")
    resources, output = args.resources.resolve(), args.output.resolve()
    env = validation_environment(repo, os.environ, cpu=True)
    env["JAX_PLATFORMS"] = "cpu"
    failures = []
    for interface, codes in recipes.items():
        for number, code in enumerate(codes, 1):
            code = code.replace("resources/", f"{resources}/").replace(
                "results/", f"{output}/{interface}/results/"
            )
            command = (
                ["bash", "-euc", code] if interface == "cli" else [sys.executable, "-c", code]
            )
            failures.append(execute(command, f"ld-{interface}-{number}", output, env))
    if any(failures):
        raise SystemExit(f"LD tutorial failures: {failures}")
    compare_outputs(output)


if __name__ == "__main__":
    main()
