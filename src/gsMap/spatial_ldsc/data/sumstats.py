"""Fast tab-separated GWAS parsing without millions of Python SNP strings.

Global QC statistics are computed *before* restricting to reference SNPs.
Unusual whitespace formats continue through the existing pandas reader.
"""

from __future__ import annotations

import bz2
import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv

logger = logging.getLogger("gsMap.spatial_ldsc.io")

# Match pandas read_csv's standard NA tokens plus gsMap's explicit "." token.
_NULL_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
    ".",
]


def read_reference_sumstats(path, reference_snps):
    """Return (reference rows, global N maximum, global lambda GC), or None.

    Arrow owns the full SNP column as compact UTF-8 buffers. Only reference
    matches become Python strings/DataFrames. Each file uses one parser thread;
    parallelism is controlled by the bounded file-preparation pool.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        header = stream.readline().rstrip("\r\n").split("\t")
    if not {"SNP", "N", "Z"}.issubset(header) or len(header) != len(set(header)):
        return None
    try:
        table = csv.read_csv(
            str(path),
            read_options=csv.ReadOptions(use_threads=False),
            parse_options=csv.ParseOptions(delimiter="\t"),
            convert_options=csv.ConvertOptions(
                include_columns=["SNP", "N", "Z"],
                column_types={"SNP": pa.string(), "N": pa.float64(), "Z": pa.float64()},
                null_values=_NULL_VALUES,
                strings_can_be_null=True,
            ),
        )
    except pa.ArrowInvalid as error:
        logger.debug("Tabular GWAS fast path unavailable for %s: %s", path, error)
        return None
    original_rows = len(table)
    # Stable first occurrence, including null SNP identifiers, just as pandas
    # drop_duplicates('SNP'). No Python set of every input SNP is constructed.
    first = pc.index_in(table["SNP"], value_set=table["SNP"])
    keep = pc.equal(first, pa.array(np.arange(original_rows, dtype=np.int32)))
    table = table.filter(keep)
    del first, keep
    n = table["N"].to_numpy().astype(np.float32)
    z = table["Z"].to_numpy().astype(np.float32)
    n_max = pd.Series(n, copy=False).max()
    lambda_gc = np.median(z**2) / 0.4559364 if len(z) else np.float32(np.nan)
    # Look up in reference order and reuse its existing Python SNP strings,
    # rather than decoding a million new strings per trait with to_pandas().
    reference = reference_snps.drop_duplicates()
    positions = pc.index_in(pa.array(reference, type=pa.string()), value_set=table["SNP"])
    positions = pc.fill_null(positions, -1).to_numpy()
    present = positions >= 0
    selected = positions[present]
    columns = {"SNP": reference[present], "N": n[selected], "Z": z[selected]}
    frame = pd.DataFrame({name: columns[name] for name in header if name in columns})
    logger.info(
        "Read %d GWAS rows (%d unique); materialized %d reference SNPs via Arrow",
        original_rows,
        len(table),
        len(frame),
    )
    return frame, n_max, lambda_gc
