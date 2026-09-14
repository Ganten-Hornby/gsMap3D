"""Shared-LD orchestration: prepare traits, dispatch chunks, publish in order.

Backend policy lives in backends.regression; bounded input scheduling lives in
execution.prefetch. The SNP union is only for LD construction: each regression
retains its own SNP order, sample sizes, filtering and jackknife blocks.
"""

from __future__ import annotations

import logging
import time
import traceback
from collections import deque
from contextlib import ExitStack, closing

import numpy as np

from gsMap.spatial_ldsc.backends.regression import SharedRegressionBackend
from gsMap.spatial_ldsc.backends.regression import select_moment_backend as _select_moments
from gsMap.spatial_ldsc.data.chunks import SpatialLDChunkSource
from gsMap.spatial_ldsc.data.loading import prepare_trait_data
from gsMap.spatial_ldsc.data.traits import PARAMETER_KEYS as PARAMETER_KEYS
from gsMap.spatial_ldsc.data.traits import prepare_shared_trait_arrays, prepare_snp_data_for_blocks
from gsMap.spatial_ldsc.data.writer import ResultWriter, result_filename
from gsMap.spatial_ldsc.execution.prefetch import prefetched_chunks as _prefetched_chunks
from gsMap.spatial_ldsc.execution.progress import LDSCProgress
from gsMap.spatial_ldsc.execution.trait_loading import bounded_trait_map, resolve_gwas_workers

logger = logging.getLogger("gsMap.spatial_ldsc.multitrait")
_INFLIGHT_PREVIOUS_BATCHES = 2


def prepare_shared_traits(engine, traits):
    """Prepare pending traits once on the host, without initializing a GPU."""
    ref = engine.ldscore_adata if engine.mode == "direct_ldscore" else engine.snp_gene_weight_adata
    rows = engine.spot_indices[engine.chunk_starts[0] : engine._cell_end]
    prepared, failures = [], []
    pending = []
    for name, path in traits:
        filename = result_filename(
            engine.config, name, engine.n_spots, int(rows[0]), int(rows[-1]) + 1
        )
        if (engine.output_dir / filename).exists():
            logger.info("Skipping completed trait %s", name)
            continue
        pending.append((name, path, filename))
    if not pending:
        return None, None, prepared, failures
    workers = resolve_gwas_workers(getattr(engine.config, "ldsc_gwas_workers", 16), len(pending))
    # Initialize lazy pandas index engines on the calling thread. Workers read
    # these shared resources, but never modify them or touch HDF5 handles.
    for table in (engine.baseline_ld, engine.w_ld):
        if table is not None:
            _ = table.index.is_unique
    _ = ref.obs_names.is_unique
    started = time.monotonic()
    logger.info(
        "Preparing %d pending GWAS traits with %d reader(s); at most %d files in flight",
        len(pending),
        workers,
        workers,
    )

    def prepare_one(indexed_item):
        ordinal, item = indexed_item
        name, path, filename = item
        try:
            data, _ = prepare_trait_data(
                engine.config, name, path, engine.baseline_ld, engine.w_ld, ref
            )
            data = prepare_snp_data_for_blocks(data, engine.config.n_blocks)
            arrays = prepare_shared_trait_arrays(
                data, name=name, filename=filename, n_ref=ref.n_obs
            )
            return ordinal, name, arrays, None
        except Exception as error:
            logger.exception("Could not prepare trait %s", name)
            # Exception tracebacks otherwise retain the full raw GWAS table in
            # failed futures until all 127 traits have finished. Keep the error
            # type/message/cause, but release traceback-local arrays.
            seen = set()
            cause = error
            while cause is not None and id(cause) not in seen:
                seen.add(id(cause))
                traceback.clear_frames(cause.__traceback__)
                cause.__traceback__ = None
                cause = cause.__cause__ or cause.__context__
            return ordinal, name, None, error

    ordered_results = [None] * len(pending)
    failed_count = 0
    with closing(
        bounded_trait_map(prepare_one, enumerate(pending), workers, ordered=False)
    ) as results:
        for index, (ordinal, name, arrays, error) in enumerate(results, 1):
            ordered_results[ordinal] = (name, arrays, error)
            failed_count += error is not None
            if index % 10 == 0 or index == len(pending):
                logger.info(
                    "Prepared GWAS traits %d/%d in %.1fs (%d failed)",
                    index,
                    len(pending),
                    time.monotonic() - started,
                    failed_count,
                )
    for name, arrays, error in ordered_results:
        if error is not None:
            failures.append((name, error))
        else:
            prepared.append(arrays)
    if not prepared:
        return None, None, prepared, failures
    union = _index_trait_union(prepared, ref.n_obs)
    sparse = None
    if engine.mode == "weight_matrix":
        sparse = engine.snp_gene_weight_adata.X[union][:, engine.weight_gene_indices].tocsr(
            copy=True
        )
        sparse.sum_duplicates()
        sparse.sort_indices()
    logger.info("Sharing %s union SNPs across %s traits", len(union), len(prepared))
    return union, sparse, prepared, failures


def _index_trait_union(prepared, n_ref):
    """Map validated reference row positions without sorting repeated trait rows.

    For many traits the reference is much smaller than concatenated positions.
    A boolean union and direct inverse lookup remove that concatenation and
    repeated binary searches. Sparse single-trait cases retain the old path.
    """
    entries = sum(len(data["positions"]) for data in prepared)
    if n_ref <= entries:
        present = np.zeros(n_ref, dtype=bool)
        for data in prepared:
            present[data["positions"]] = True
        union = np.flatnonzero(present)
        del present
        inverse = np.empty(n_ref, dtype=np.int32)
        inverse[union] = np.arange(len(union), dtype=np.int32)
        for data in prepared:
            data["indices"] = inverse[data.pop("positions")]
    else:
        union = np.unique(np.concatenate([data["positions"] for data in prepared]))
        for data in prepared:
            data["indices"] = np.searchsorted(union, data.pop("positions")).astype(np.int32)
    return union


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


def _abort_if_open(writer: ResultWriter) -> None:
    if not writer.closed:
        writer.abort()


def _wait_inflight(inflight: deque) -> None:
    """Hold producer owners until asynchronous regression consumers finish."""
    for _, _, batch in inflight:
        batch.wait()


def _write_batch(item: tuple, writers: list[ResultWriter]) -> int:
    index, chunk, batch = item
    result = batch.to_host()
    for k, writer in enumerate(writers):
        writer.add_chunk_result(index, result[k, 0], result[k, 1], chunk[1], chunk[2], chunk[3])
    return len(chunk[1])


def run_shared_chunks(
    source,
    config,
    prepared,
    sparse,
    output_dir,
    n_spots,
    *,
    emit_header=True,
    progress_callback=None,
):
    """Run a bounded reader → regression → per-trait writer pipeline.

    Up to two prior batches remain in flight, retaining their DLPack owners.
    Readers are joined and consumers drained before backends/sources close,
    including on preparation, read, compute or write failure.
    """
    started = time.monotonic()
    writers = []
    inflight = deque()
    with ExitStack() as resources:
        resources.callback(source.close_sequential_marker_access)
        backend = resources.enter_context(
            SharedRegressionBackend(
                config,
                sparse,
                prepared,
                len(source.ldscore_snp_positions),
                moment_selector=_select_moments,
            )
        )
        logger.info("Shared regression backend: %s", backend.name)
        expected = source.cell_end - source.chunk_starts[0]
        progress = LDSCProgress(
            expected,
            len(prepared),
            len(source.chunk_starts),
            started=started,
            callback=progress_callback,
        )
        completed_cells = completed_chunks = 0
        for data in prepared:
            writer = ResultWriter(
                config,
                data["name"],
                output_dir,
                n_spots,
                len(source.chunk_starts),
                expected,
                emit_header=emit_header,
            )
            writers.append(writer)
            resources.callback(_abort_if_open, writer)
        # ExitStack runs these in reverse: join readers, restore thread limits,
        # drain consumers, abort incomplete writers, close backends, close source.
        resources.callback(_wait_inflight, inflight)
        resources.enter_context(backend.thread_context())
        chunks = resources.enter_context(closing(_prefetched_chunks(source, config)))
        for i, chunk in enumerate(chunks):
            for writer in writers:
                writer.check_errors()
            batch = backend.compute(
                marker=chunk[4] if sparse is not None else None, ldscore=chunk[0]
            )
            inflight.append((i, chunk, batch))
            if len(inflight) > _INFLIGHT_PREVIOUS_BATCHES:
                completed_cells += _write_batch(inflight.popleft(), writers)
                completed_chunks += 1
            progress.update(completed_cells, completed_chunks, dispatched=i + 1)
        while inflight:
            completed_cells += _write_batch(inflight.popleft(), writers)
            completed_chunks += 1
            progress.update(completed_cells, completed_chunks, dispatched=len(source.chunk_starts))
        paths = [writer.finalize() for writer in writers]
        progress.update(completed_cells, completed_chunks, force=True)
        seconds = time.monotonic() - started
        logger.info(
            "Shared LDSC completed %d cells × %d traits in %.2fs (%.1f cell-traits/s, including setup and writing)",
            expected,
            len(prepared),
            seconds,
            expected * len(prepared) / seconds,
        )
        return paths


def run_shared_engine(engine, traits):
    union, sparse, prepared, failures = prepare_shared_traits(engine, traits)
    if prepared:
        if engine.config.ldsc_num_gpus > 1:
            from gsMap.spatial_ldsc.execution.multi_gpu import run_multi_gpu

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
