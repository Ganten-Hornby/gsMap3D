#!/usr/bin/env python3
"""Prepare full-data, human-gene, per-section E11.5 trait-mapping inputs."""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from gsMap.find_latent.st_process import convert_to_human_genes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--homolog-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--section-key", default="slices")
    parser.add_argument("--z-key", default="aligned_spatial_3D_mm")
    return parser.parse_args()


def ordered_sections(adata: ad.AnnData, section_key: str, z_key: str) -> list[str]:
    sections = adata.obs[section_key].astype(str).to_numpy()
    z = np.asarray(adata.obsm[z_key])[:, 2]
    unique_sections = np.unique(sections)
    return sorted(
        unique_sections.tolist(),
        key=lambda section: float(np.median(z[sections == section])),
    )


def outputs_are_complete(manifest: Path, audit: Path) -> bool:
    if not manifest.exists() or not audit.exists():
        return False
    with manifest.open() as handle:
        outputs = yaml.safe_load(handle)
    return bool(outputs) and all(Path(path).is_file() for path in outputs.values())


def prepare_inputs(args: argparse.Namespace) -> None:
    if outputs_are_complete(args.manifest, args.audit):
        print(f"Validated existing complete input manifest: {args.manifest}")
        return

    homologs = pd.read_csv(args.homolog_file, sep="\t")
    required_columns = {"MOUSE_GENE_SYM", "HUMAN_GENE_SYM"}
    if not required_columns.issubset(homologs.columns):
        raise ValueError(
            f"Homolog file must contain {sorted(required_columns)}; got {homologs.columns.tolist()}"
        )
    if homologs["MOUSE_GENE_SYM"].duplicated().any():
        raise ValueError("Homolog file contains duplicated mouse gene symbols")
    mapping = dict(
        zip(
            homologs["MOUSE_GENE_SYM"].astype(str),
            homologs["HUMAN_GENE_SYM"].astype(str),
            strict=True,
        )
    )

    adata = ad.read_h5ad(args.source)
    required_obs = {args.section_key, "heart_regions", "ctype_user"}
    missing_obs = required_obs.difference(adata.obs.columns)
    missing_obsm = {args.z_key, "emb_cell", "emb_niche"}.difference(adata.obsm)
    if missing_obs or missing_obsm:
        raise ValueError(f"Missing obs={sorted(missing_obs)} or obsm={sorted(missing_obsm)}")

    source_n_obs = int(adata.n_obs)
    source_n_vars = int(adata.n_vars)
    mapped_source_genes = sum(gene in mapping for gene in adata.var_names)
    if mapped_source_genes < 300:
        raise ValueError(
            f"Only {mapped_source_genes} source genes map to human symbols; stopping"
        )
    adata = convert_to_human_genes(adata, mapping, species="MOUSE")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    sections = ordered_sections(adata, args.section_key, args.z_key)
    section_values = adata.obs[args.section_key].astype(str).to_numpy()
    z = np.asarray(adata.obsm[args.z_key])[:, 2]
    manifest: dict[str, str] = {}
    section_audit: list[dict[str, object]] = []

    for rank, section in enumerate(sections):
        mask = section_values == section
        sample_name = f"Mouse_E11.5_heart_slice_{rank:03d}"
        output = (args.output_dir / f"{sample_name}.h5ad").resolve()
        temporary = output.with_suffix(".h5ad.tmp")
        subset = adata[mask].copy()
        subset.write_h5ad(temporary)
        temporary.replace(output)
        manifest[sample_name] = str(output)
        section_audit.append(
            {
                "rank": rank,
                "section": section,
                "sample_name": sample_name,
                "n_obs": int(mask.sum()),
                "median_z_mm": float(np.median(z[mask])),
                "output": str(output),
            }
        )

    manifest_tmp = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    with manifest_tmp.open("w") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    manifest_tmp.replace(args.manifest)

    audit_data = {
        "source": str(args.source.resolve()),
        "homolog_file": str(args.homolog_file.resolve()),
        "source_n_obs": source_n_obs,
        "source_n_vars": source_n_vars,
        "mapped_source_genes": int(mapped_source_genes),
        "human_n_vars_after_deduplication": int(adata.n_vars),
        "n_sections": len(sections),
        "output_n_obs": int(sum(item["n_obs"] for item in section_audit)),
        "no_downsampling": True,
        "section_key": args.section_key,
        "z_key": args.z_key,
        "sections": section_audit,
        "manifest": str(args.manifest.resolve()),
    }
    audit_tmp = args.audit.with_suffix(args.audit.suffix + ".tmp")
    audit_tmp.write_text(json.dumps(audit_data, indent=2) + "\n")
    audit_tmp.replace(args.audit)
    print(
        f"Prepared {source_n_obs:,} observations across {len(sections)} sections "
        f"with {adata.n_vars:,} human genes"
    )


def main() -> None:
    prepare_inputs(parse_args())


if __name__ == "__main__":
    main()
