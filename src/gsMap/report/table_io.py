"""Fast report metadata exports; the browser continues to consume JavaScript."""

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as arrow_csv
import pyarrow.parquet as parquet

logger = logging.getLogger(__name__)
_SIGNATURE = b"gsmap_source_csv"


def _signature(path: Path) -> dict:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def write_metadata(frame: pd.DataFrame, csv_path: Path) -> None:
    """Keep a portable CSV and a validated, lossless Parquet companion.

    CSV is uncompressed. Arrow avoids pandas' per-value Python formatting.
    A companion is accepted only while the source CSV fingerprint is unchanged.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    logger.info(f"Metadata Arrow conversion: {time.perf_counter() - started:.2f}s")
    for destination, is_csv in ((csv_path, True), (csv_path.with_suffix(".parquet"), False)):
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(descriptor)
        started = time.perf_counter()
        try:
            if is_csv:
                arrow_csv.write_csv(table, temporary)
            else:
                metadata = dict(table.schema.metadata or {})
                metadata[_SIGNATURE] = json.dumps(_signature(csv_path)).encode()
                parquet.write_table(
                    table.replace_schema_metadata(metadata),
                    temporary,
                    compression="snappy",
                    row_group_size=250_000,
                )
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        logger.info(
            f"Metadata {destination.suffix} export: {time.perf_counter() - started:.2f}s; "
            f"{destination.stat().st_size / 1024**3:.2f} GiB"
        )


def read_metadata_companion(csv_path: Path, *, labels=()) -> pd.DataFrame | None:
    """Return a fresh companion, otherwise let the caller use its CSV reader."""
    csv_path = Path(csv_path)
    companion = csv_path.with_suffix(".parquet")
    if not companion.is_file() or not csv_path.is_file():
        return None
    started = time.perf_counter()
    try:
        schema = parquet.read_schema(companion)
        recorded = json.loads((schema.metadata or {}).get(_SIGNATURE, b"null"))
        if recorded != _signature(csv_path):
            return None
        frame = pd.read_parquet(companion)
        # Match the CSV reader's identifier contract: keep "01"/"NA" as labels;
        # only empty/missing labels become missing.
        for label in labels:
            if label in frame:
                frame[label] = frame[label].map(
                    lambda value: str(value) if pd.notna(value) and str(value) else np.nan
                )
        logger.info(f"Read metadata Parquet companion in {time.perf_counter() - started:.2f}s")
        return frame
    except (OSError, ValueError, pa.ArrowException) as error:
        logger.warning(f"Cannot read metadata companion {companion}; using CSV: {error}")
        return None
