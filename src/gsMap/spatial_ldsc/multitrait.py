"""Shared SNP-union LD construction with trait-specific regressions.

The union is used only for sparse multiplication. Every regression retains its
own SNP order, sample sizes, filtering, baseline annotations and jackknife blocks.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np

from .chunk_source import SpatialLDChunkSource
from .io import prepare_trait_data
from .result_writer import ResultWriter, result_filename
from .trait_data import prepare_snp_data_for_blocks

logger = logging.getLogger("gsMap.spatial_ldsc.multitrait")
PARAMETER_KEYS = ("baseline_sum", "chisq", "N", "baseline", "w_ld", "Nbar")


def _select_moments(config, sparse, prepared, ld_backend, platform):
    """Choose a supported fast path without hiding native execution failures."""
    from .moment_design import supports_moments

    requested = getattr(config, "ldsc_regression_backend", "auto")
    if requested not in ("auto", "jax", "moments"):
        raise ValueError("Unknown LDSC regression backend")
    if not 1 <= config.ldsc_trait_batch_size <= 16:
        raise ValueError("ldsc_trait_batch_size must be 1..16")
    if requested == "jax":
        return None
    if supports_moments(sparse, prepared):
        if platform == "cpu":
            from .moment_cpu import CPUMomentRegression

            return CPUMomentRegression(
                sparse,
                prepared,
                config.n_blocks,
                threads=getattr(config, "ldsc_compute_workers", 10),
                trait_batch=config.ldsc_trait_batch_size,
            )
        if ld_backend is not None and hasattr(ld_backend, "compute_torch"):
            try:
                from .moment_gpu import GPUMomentRegression
            except ImportError:
                logger.warning("Optional Triton moment backend unavailable; using JAX")
            else:
                return GPUMomentRegression(
                    sparse,
                    prepared,
                    config.n_blocks,
                    ld_backend,
                    trait_batch=config.ldsc_trait_batch_size,
                )
    if requested == "moments":
        raise ValueError(
            "Moments require float32 CSR marker mode, two baseline annotations "
            "and a supported CPU or CUDA/Triton backend"
        )
    logger.info("Using general JAX regression for this input/backend")
    return None


def _prefetched_chunks(source, config):
    # Array reads have no mutable reader cursor. HDF5 and file-reader wrappers
    # keep a single worker; never issue concurrent reads on a shared HDF5 handle.
    marker = getattr(source, "marker_score_adata", None)
    parallel = marker is not None and isinstance(marker.X, np.ndarray)
    workers = min(4, max(1, getattr(config, "ldsc_read_workers", 4))) if parallel else 1
    depth = 2 * workers
    count = len(source.chunk_starts)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ldsc-shared-reader") as pool:
        futures = deque()
        next_chunk = 0
        try:
            while next_chunk < min(depth, count):
                futures.append(pool.submit(source.fetch_chunk, next_chunk, True))
                next_chunk += 1
            while futures:
                chunk = futures.popleft().result()
                if next_chunk < count:
                    futures.append(pool.submit(source.fetch_chunk, next_chunk, True))
                    next_chunk += 1
                yield chunk
        finally:
            for future in futures:
                future.cancel()


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
    Four-column marker models use fused moments on CPU/CUDA when available.
    Custom baselines and direct LD scores retain the general JAX regression.
    """
    import jax
    import jax.numpy as jnp

    from .ldscore_backend import select_gpu_ldscore_backend
    from .multitrait_kernel import mean_regression, prepare_mean_design, shared_hsq

    backend = None
    moments = None
    writers = []
    inflight = deque()
    started = time.monotonic()
    try:
        if sparse is not None and jax.default_backend() == "gpu":
            backend = select_gpu_ldscore_backend(sparse)
        moments = _select_moments(config, sparse, prepared, backend, jax.default_backend())
        cpu_moments = moments is not None and jax.default_backend() == "cpu"
        params = (
            []
            if moments is not None
            else [
                (jnp.asarray(d["indices"]), tuple(jnp.asarray(d[key]) for key in PARAMETER_KEYS))
                for d in prepared
            ]
        )
        design = (
            None
            if moments is not None
            else prepare_mean_design(
                len(source.ldscore_snp_positions),
                [d["indices"] for d in prepared],
                [d["N"] for d in prepared],
                [d["baseline_sum"] for d in prepared],
                [d["chisq"] for d in prepared],
            )
        )
        group_size = config.ldsc_trait_batch_size
        groups = [
            (i, tuple(params[i : i + group_size])) for i in range(0, len(params), group_size)
        ]
        logger.info("Shared regression backend: %s", type(moments).__name__ if moments else "JAX")
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
            if moments is not None and (
                not np.isfinite(result).all() or np.any(result[:, 1] <= 0)
            ):
                raise FloatingPointError("Singular or invalid spatial LDSC moment regression")
            for k, writer in enumerate(writers):
                writer.add_chunk_result(
                    index, result[k, 0], result[k, 1], chunk[1], chunk[2], chunk[3]
                )

        last_log = started
        # The projection is small; avoid BLAS oversubscription alongside Numba.
        if cpu_moments:
            from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api="blas") if cpu_moments else nullcontext():
            for i, chunk in enumerate(_prefetched_chunks(source, config)):
                for writer in writers:
                    writer.check_errors()
                if moments is not None:
                    if cpu_moments:
                        results, owner = [moments.compute(chunk[4])], None
                    else:
                        results, owner = moments.compute(chunk[4])
                else:
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
                    owner = ld
                inflight.append((i, chunk, results, owner))
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
            if moments is not None:
                moments.close()
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
