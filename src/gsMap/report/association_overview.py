"""Compact full-result annotation distributions, independent of map sampling.

Histograms use a common trait axis for every section/category. The browser only
smooths these histograms for display; Cauchy statistics are joined from existing
results, never estimated from the histogram or combined across sections here.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
OVERVIEW_VERSION = 1
N_BINS = 96


def _labels(series):
    return series.astype("string").fillna("NaN").astype(str)


def export_association_overview(metadata, report_data_dir, js_data_dir, traits, annotations):
    """Write lazy per-trait/annotation/scope JS shards and an atomic index.

    Uses all metadata rows, including spots without a finite trait score in the
    reported spot count. Additional memory is O(n_spots + n_categories * bins),
    rather than one large spot-by-trait or sample-by-category-by-bin tensor.
    """
    from .report_data import _atomic_write_text, _read_report_csv, _report_file_signature

    report_data_dir, js_data_dir = Path(report_data_dir), Path(js_data_dir)
    source = report_data_dir / "spot_metadata.csv"
    cauchy_path = report_data_dir / "cauchy_results.csv"
    manifest_path = js_data_dir / "association_overview_manifest.json"
    signature = {
        "version": OVERVIEW_VERSION,
        "summary_revision": 3,
        "n_bins": N_BINS,
        "metadata": _report_file_signature(source),
        "cauchy": _report_file_signature(cauchy_path) if cauchy_path.is_file() else None,
        "traits": list(traits),
        "annotations": list(annotations),
    }
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest["inputs"] == signature and all(
            (js_data_dir / path).is_file() and _report_file_signature(js_data_dir / path) == stat
            for path, stat in manifest["outputs"].items()
        ):
            return manifest["index"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    required = {"sample_name", *traits, *annotations}
    if missing := required.difference(metadata.columns):
        raise ValueError(f"Association overview missing columns: {sorted(missing)}")
    if not len(metadata):
        raise ValueError("Association overview requires at least one spot")

    cauchy = {}
    if cauchy_path.is_file():
        frame = _read_report_csv(
            cauchy_path, labels=["trait", "annotation_name", "annotation", "sample_name", "type"]
        )
        for row in frame.to_dict("records"):
            kind = row.get("type")
            if kind not in ("aggregated", "sample"):
                continue
            sample = None if kind == "aggregated" else str(row.get("sample_name"))
            key = (str(row["trait"]), str(row["annotation_name"]), sample, str(row["annotation"]))
            if key in cauchy:
                raise ValueError(f"Ambiguous duplicate Cauchy summary: {key}")
            n_spots = row.get("total_spots")
            cauchy[key] = {
                "n_spots": int(n_spots) if pd.notna(n_spots) and np.isfinite(n_spots) else None,
            }
            for field, source_field in (
                ("cauchy_logp", "mlog10_p_cauchy"),
                ("median_logp", "mlog10_p_median"),
                ("q95_logp", "top_95_quantile"),
            ):
                value = row.get(source_field)
                cauchy[key][field] = (
                    float(value) if pd.notna(value) and np.isfinite(value) else None
                )

    sample_groups = (
        _labels(metadata["sample_name"])
        .groupby(_labels(metadata["sample_name"]), sort=False)
        .indices
    )
    index = {"version": OVERVIEW_VERSION, "source_n_spots": len(metadata), "traits": {}}
    outputs = {}
    for annotation in annotations:
        codes, categories = pd.factorize(_labels(metadata[annotation]), sort=True)
        n_categories = len(categories)
        for trait in traits:
            values = pd.to_numeric(metadata[trait], errors="raise").to_numpy(dtype=float)
            if np.isinf(values).any() or (values < 0).any():
                raise ValueError(
                    f"Overview trait {trait!r} must contain nonnegative scores or NaN"
                )
            valid = np.isfinite(values)
            maximum = max(float(values[valid].max()) if valid.any() else 0.0, 1.0)
            bins = np.minimum(
                (np.where(valid, values, 0) / maximum * N_BINS).astype(int), N_BINS - 1
            )
            entries = {"samples": {}}
            index["traits"].setdefault(trait, {})[annotation] = entries
            for sample, positions in [(None, slice(None)), *sample_groups.items()]:
                local_codes, local_valid = codes[positions], valid[positions]
                count = np.bincount(local_codes, minlength=n_categories)
                scored_codes = local_codes[local_valid]
                histograms = np.bincount(
                    scored_codes * N_BINS + bins[positions][local_valid],
                    minlength=n_categories * N_BINS,
                ).reshape(n_categories, N_BINS)
                rows = []
                for code in np.flatnonzero(count):
                    category = str(categories[code])
                    scored = int(histograms[code].sum())
                    summary = cauchy.get((trait, annotation, sample, category), {})
                    rows.append(
                        {
                            "category": category,
                            "n_spots": int(count[code]),
                            "n_scored": scored,
                            "cauchy_logp": summary.get("cauchy_logp"),
                            "median_logp": summary.get("median_logp"),
                            "q95_logp": summary.get("q95_logp"),
                            "cauchy_n_spots": summary.get("n_spots"),
                            "histogram": histograms[code].tolist(),
                        }
                    )
                payload = {
                    "version": OVERVIEW_VERSION,
                    "trait": trait,
                    "annotation": annotation,
                    "sample": sample,
                    "n_spots": int(count.sum()),
                    "n_bins": N_BINS,
                    "x_max": maximum,
                    "rows": rows,
                    "sampling": "all",
                }
                token = hashlib.sha256(
                    json.dumps([trait, annotation, sample]).encode()
                ).hexdigest()[:24]
                # Content-addressed files prevent an old index from seeing a newer partial build.
                encoded = json.dumps(
                    payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")
                )
                digest = hashlib.sha256(encoded.encode()).hexdigest()[:16]
                key = f"{token}_{digest}"
                relative = f"association_overview/{key}.js"
                path = js_data_dir / relative
                _atomic_write_text(
                    path,
                    f"window.GSMAP_ASSOCIATION_OVERVIEW_DATA ||= {{}};\nwindow.GSMAP_ASSOCIATION_OVERVIEW_DATA[{json.dumps(key)}]={encoded};\n",
                )
                outputs[relative] = _report_file_signature(path)
                entry = {"file": f"js_data/{relative}", "key": key}
                if sample is None:
                    entries["all"] = entry
                else:
                    entries["samples"][sample] = entry
    index_path = js_data_dir / "association_overview_index.js"
    _atomic_write_text(
        index_path,
        f"window.GSMAP_ASSOCIATION_OVERVIEW_INDEX={json.dumps(index, ensure_ascii=True, allow_nan=False)};\n",
    )
    outputs[index_path.name] = _report_file_signature(index_path)
    _atomic_write_text(
        manifest_path, json.dumps({"inputs": signature, "outputs": outputs, "index": index})
    )
    logger.info(
        f"Association overview: {len(metadata):,} full-result spots; {len(outputs) - 1} lazy shards"
    )
    return index
