"""Shared SNP-union LD construction with trait-specific regressions.

The union is used only for sparse multiplication. Every regression retains its
own SNP order, sample sizes, filtering, baseline annotations and jackknife blocks.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .chunk_source import SpatialLDChunkSource
from .io import prepare_trait_data
from .result_writer import ResultWriter, result_filename
from .trait_data import prepare_snp_data_for_blocks

logger = logging.getLogger("gsMap.spatial_ldsc.multitrait")
PARAMETER_KEYS = ("baseline_sum", "chisq", "N", "baseline", "w_ld", "Nbar")


def prepare_shared_traits(engine, traits):
    """Prepare pending traits once on the host, without initializing a GPU."""
    ref = engine.ldscore_adata if engine.mode == "direct_ldscore" else engine.snp_gene_weight_adata
    rows = engine.spot_indices[engine.chunk_starts[0] : engine._cell_end]
    prepared, failures = [], []
    for name, path in traits:
        filename = result_filename(
            engine.config, name, engine.n_spots, int(rows[0]), int(rows[-1]) + 1
        )
        if (engine.output_dir / filename).exists():
            logger.info("Skipping completed trait %s", name)
            continue
        try:
            data, _ = prepare_trait_data(
                engine.config, name, path, engine.baseline_ld, engine.w_ld, ref
            )
            data = prepare_snp_data_for_blocks(data, engine.config.n_blocks)
            positions = np.asarray(data["snp_positions"], dtype=np.int64)
            if np.any(positions < 0) or np.any(positions >= ref.n_obs):
                raise ValueError("Trait SNPs are missing from the LD reference")
            n = np.asarray(data["N"], dtype=np.float32)
            nbar = np.float32(data["Nbar"])  # original mean BEFORE block truncation
            baseline = data["baseline_ld"].values.astype(np.float32) * n[:, None] / nbar
            baseline = np.concatenate([baseline, np.ones((len(n), 1), dtype=np.float32)], axis=1)
            prepared.append(
                dict(
                    name=name,
                    filename=filename,
                    positions=positions,
                    baseline_sum=np.asarray(data["baseline_ld_sum"], dtype=np.float32),
                    chisq=np.asarray(data["chisq"], dtype=np.float32),
                    N=n,
                    Nbar=nbar,
                    baseline=baseline,
                    w_ld=np.asarray(data["w_ld"], dtype=np.float32),
                )
            )
        except Exception as error:
            logger.exception("Could not prepare trait %s", name)
            failures.append((name, error))
    if not prepared:
        return None, None, prepared, failures
    union = np.unique(np.concatenate([d["positions"] for d in prepared]))
    for data in prepared:
        data["indices"] = np.searchsorted(union, data.pop("positions")).astype(np.int32)
    sparse = None
    if engine.mode == "weight_matrix":
        sparse = engine.snp_gene_weight_adata.X[union][:, engine.weight_gene_indices].tocsr(
            copy=True
        )
        sparse.sum_duplicates()
        sparse.sort_indices()
    logger.info("Sharing %s union SNPs across %s traits", len(union), len(prepared))
    return union, sparse, prepared, failures


def make_shared_source(engine, union):
    return SpatialLDChunkSource(
        mode=engine.mode,
        chunk_starts=engine.chunk_starts,
        chunk_size=engine.chunk_size,
        cell_end=engine._cell_end,
        spot_indices=engine.spot_indices,
        spot_names_filtered=engine.spot_names_filtered,
        ldscore_adata=engine.ldscore_adata,
        ldscore_snp_positions=union,
        marker_score_adata=engine.marker_score_adata,
        marker_score_gene_indices=engine.marker_score_gene_indices,
    )


def run_shared_chunks(source, config, prepared, sparse, output_dir, n_spots, *, emit_header=True):
    """Bounded reader → shared LD → grouped JAX → per-trait ordered writers.

    Two prior batches may remain in flight; each owns its DLPack LD array until
    all consumers finish. Writer failures invalidate the run and discard partials.
    CPU support here is for numerical tests; the public CPU route stays unchanged.
    """
    import jax
    import jax.numpy as jnp

    from .ldscore_backend import select_gpu_ldscore_backend
    from .multitrait_kernel import mean_regression, prepare_mean_design, shared_hsq

    backend = None
    writers = []
    inflight = deque()
    started = time.monotonic()
    try:
        params = [
            (jnp.asarray(d["indices"]), tuple(jnp.asarray(d[key]) for key in PARAMETER_KEYS))
            for d in prepared
        ]
        design = prepare_mean_design(
            len(source.ldscore_snp_positions),
            [d["indices"] for d in prepared],
            [d["N"] for d in prepared],
            [d["baseline_sum"] for d in prepared],
            [d["chisq"] for d in prepared],
        )
        group_size = config.ldsc_trait_batch_size
        groups = [
            (i, tuple(params[i : i + group_size])) for i in range(0, len(params), group_size)
        ]
        if sparse is not None and jax.default_backend() == "gpu":
            backend = select_gpu_ldscore_backend(sparse)
        expected = source.cell_end - source.chunk_starts[0]
        for data in prepared:
            writers.append(
                ResultWriter(
                    config,
                    data["name"],
                    output_dir,
                    n_spots,
                    len(source.chunk_starts),
                    expected,
                    emit_header=emit_header,
                )
            )

        def finish(item):
            index, chunk, results, owner = item
            result = np.concatenate(jax.device_get(results), axis=0)
            for k, writer in enumerate(writers):
                writer.add_chunk_result(
                    index, result[k, 0], result[k, 1], chunk[1], chunk[2], chunk[3]
                )

        last_log = started
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ldsc-shared-reader") as pool:
            future = pool.submit(source.fetch_chunk, 0, True)
            for i in range(len(source.chunk_starts)):
                for writer in writers:
                    writer.check_errors()
                chunk = future.result()
                if i + 1 < len(source.chunk_starts):
                    future = pool.submit(source.fetch_chunk, i + 1, True)
                if sparse is None:
                    ld = jnp.asarray(chunk[0])
                elif backend is not None:
                    ld = backend.compute(chunk[4])
                else:
                    ld = jnp.asarray(sparse @ chunk[4])
                hsq = shared_hsq(ld, *design)
                results = [
                    mean_regression(config.n_blocks, ld, group, hsq[lo : lo + len(group)])
                    for lo, group in groups
                ]
                inflight.append((i, chunk, results, ld))
                if len(inflight) > 2:
                    finish(inflight.popleft())
                now = time.monotonic()
                if now - last_log >= 30:
                    logger.info(
                        "Shared LDSC dispatched %d/%d chunks (%d traits)",
                        i + 1,
                        len(source.chunk_starts),
                        len(prepared),
                    )
                    last_log = now
            while inflight:
                finish(inflight.popleft())
        paths = [writer.finalize() for writer in writers]
        seconds = time.monotonic() - started
        logger.info(
            "Shared LDSC completed %d cells × %d traits in %.2fs (%.1f cell-traits/s, including setup and writing)",
            expected,
            len(prepared),
            seconds,
            expected * len(prepared) / seconds,
        )
        return paths
    finally:
        for writer in writers:
            if not writer.closed:
                writer.abort()
        # Even on output/read failure, asynchronous consumers must finish before
        # their DLPack producers can be released.
        try:
            for item in inflight:
                jax.block_until_ready(item[2])
        finally:
            source.close_sequential_marker_access()
            if backend is not None:
                backend.close()


def run_shared_engine(engine, traits):
    union, sparse, prepared, failures = prepare_shared_traits(engine, traits)
    if prepared:
        if engine.config.ldsc_num_gpus > 1:
            from .multi_gpu import run_multi_gpu

            run_multi_gpu(engine, union, sparse, prepared)
        else:
            run_shared_chunks(
                make_shared_source(engine, union),
                engine.config,
                prepared,
                sparse,
                engine.output_dir,
                engine.n_spots,
            )
    if failures:
        raise RuntimeError(
            "Spatial LDSC failed to prepare traits: " + ", ".join(name for name, _ in failures)
        ) from failures[0][1]
