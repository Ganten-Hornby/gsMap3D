"""Shared-LD orchestration: prepare traits, dispatch chunks, publish in order.

Numerical/backend policy lives in regression_backend; bounded input scheduling
lives in prefetch. The SNP union is only for LD construction: each regression
retains its own SNP order, sample sizes, filtering and jackknife blocks.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import ExitStack, closing

import numpy as np

from .chunk_source import SpatialLDChunkSource
from .io import prepare_trait_data
from .prefetch import prefetched_chunks as _prefetched_chunks
from .regression_backend import SharedRegressionBackend
from .regression_backend import select_moment_backend as _select_moments
from .result_writer import ResultWriter, result_filename
from .trait_data import PARAMETER_KEYS as PARAMETER_KEYS
from .trait_data import prepare_shared_trait_arrays, prepare_snp_data_for_blocks

logger = logging.getLogger("gsMap.spatial_ldsc.multitrait")
_INFLIGHT_PREVIOUS_BATCHES = 2


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
            prepared.append(
                prepare_shared_trait_arrays(data, name=name, filename=filename, n_ref=ref.n_obs)
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


def _abort_if_open(writer: ResultWriter) -> None:
    if not writer.closed:
        writer.abort()


def _wait_inflight(inflight: deque) -> None:
    """Hold producer owners until asynchronous regression consumers finish."""
    for _, _, batch in inflight:
        batch.wait()


def _write_batch(item: tuple, writers: list[ResultWriter]) -> None:
    index, chunk, batch = item
    result = batch.to_host()
    for k, writer in enumerate(writers):
        writer.add_chunk_result(index, result[k, 0], result[k, 1], chunk[1], chunk[2], chunk[3])


def run_shared_chunks(source, config, prepared, sparse, output_dir, n_spots, *, emit_header=True):
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
        last_log = started
        for i, chunk in enumerate(chunks):
            for writer in writers:
                writer.check_errors()
            batch = backend.compute(
                marker=chunk[4] if sparse is not None else None, ldscore=chunk[0]
            )
            inflight.append((i, chunk, batch))
            if len(inflight) > _INFLIGHT_PREVIOUS_BATCHES:
                _write_batch(inflight.popleft(), writers)
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
            _write_batch(inflight.popleft(), writers)
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
