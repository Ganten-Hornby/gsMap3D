import argparse
import gc
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


SHAPE = (6_993_667, 15_201)
DTYPE = np.dtype(np.float16)
CHUNK_ROWS = 256


def load_memmap_dense(project_root: Path):
    module_path = project_root / "src/gsMap/latent2gene/memmap_io.py"
    spec = importlib.util.spec_from_file_location("gsmap_memmap_io_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MemMapDense


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n")
    temporary_path.replace(path)


def max_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.probe_dir.mkdir(parents=True, exist_ok=True)
    memmap_path = args.probe_dir / "marker_scores"
    data_path = memmap_path.with_suffix(".dat")
    meta_path = memmap_path.with_suffix(".meta.json")
    byte_size = int(np.prod(SHAPE, dtype=np.int64)) * DTYPE.itemsize
    starts = [0, SHAPE[0] // 4, SHAPE[0] // 2, SHAPE[0] - CHUNK_ROWS]
    values = [1.0, 2.0, 3.0, 4.0]
    payload = {
        "status": "running",
        "shape": list(SHAPE),
        "dtype": DTYPE.name,
        "expected_bytes": byte_size,
        "expected_gib": byte_size / 1024**3,
        "concurrent_workers": min(args.workers, len(starts)),
        "chunk_rows": CHUNK_ROWS,
        "probe_starts": starts,
        "full_sequential_write": False,
        "started_at_unix": time.time(),
    }
    write_report(args.report, payload)

    MemMapDense = load_memmap_dense(args.project_root)
    writer = None
    reader = None
    try:
        allocation_started = time.monotonic()
        writer = MemMapDense(
            memmap_path,
            shape=SHAPE,
            dtype=DTYPE,
            mode="w",
            num_write_workers=0,
        )
        payload["allocation_seconds"] = time.monotonic() - allocation_started

        file_stat = data_path.stat()
        allocated_bytes = file_stat.st_blocks * 512
        stripe_count = int(
            subprocess.check_output(
                ["lfs", "getstripe", "-c", str(data_path)], text=True
            ).strip()
        )
        payload.update(
            {
                "actual_size_bytes": file_stat.st_size,
                "allocated_bytes": allocated_bytes,
                "allocated_fraction": allocated_bytes / byte_size,
                "stripe_count": stripe_count,
            }
        )
        assert file_stat.st_size == byte_size
        assert allocated_bytes >= byte_size
        assert stripe_count == 4

        def write_chunk(start: int, value: float) -> None:
            stop = start + CHUNK_ROWS
            chunk = np.full((CHUNK_ROWS, SHAPE[1]), value, dtype=DTYPE)
            writer.memmap[start:stop] = chunk

        write_started = time.monotonic()
        with ThreadPoolExecutor(max_workers=min(args.workers, len(starts))) as executor:
            futures = [
                executor.submit(write_chunk, start, value)
                for start, value in zip(starts, values, strict=True)
            ]
            for future in futures:
                future.result()
        writer.memmap.flush()
        with open(data_path, "r+b") as backing_file:
            os.fsync(backing_file.fileno())
        payload["concurrent_write_and_flush_seconds"] = time.monotonic() - write_started

        writer.close()
        del writer.memmap
        writer = None
        gc.collect()

        read_started = time.monotonic()
        reader = MemMapDense(
            memmap_path,
            shape=SHAPE,
            dtype=DTYPE,
            mode="r",
            num_write_workers=0,
        )
        verified = []
        for start, value in zip(starts, values, strict=True):
            observed = reader.read_batch(slice(start, start + CHUNK_ROWS))
            if not np.all(observed == np.float16(value)):
                raise AssertionError(f"Read-back mismatch at rows {start}:{start + CHUNK_ROWS}")
            verified.append(
                {
                    "start": start,
                    "stop": start + CHUNK_ROWS,
                    "value": value,
                }
            )
        payload["remap_and_read_seconds"] = time.monotonic() - read_started
        payload["verified_chunks"] = verified

        del reader.memmap
        reader = None
        gc.collect()
        data_path.unlink()
        meta_path.unlink()
        payload["cleanup_verified"] = not data_path.exists() and not meta_path.exists()
        payload["max_rss_gib"] = max_rss_gib()
        payload["status"] = "passed"
        payload["finished_at_unix"] = time.time()
        write_report(args.report, payload)
    except BaseException as error:
        payload["status"] = "failed"
        payload["error"] = repr(error)
        payload["traceback"] = traceback.format_exc()
        payload["max_rss_gib"] = max_rss_gib()
        payload["finished_at_unix"] = time.time()
        write_report(args.report, payload)
        raise


if __name__ == "__main__":
    main()
