"""Throughput counts synchronized cells × actual traits, not padded dispatches."""

import logging
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from gsMap.spatial_ldsc.execution import progress, shared
from gsMap.spatial_ldsc.execution.multi_gpu import wait_workers


def test_progress_rate_tail_and_interval(monkeypatch, caplog):
    clock = [10.0]
    monkeypatch.setattr(progress, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    records = []
    reporter = progress.LDSCProgress(1037, 4, 11, callback=records.append)
    with caplog.at_level(logging.INFO):
        reporter.update(100, 1, dispatched=3)
        assert not records
        clock[0] = 40.0
        reporter.update(800, 8, dispatched=10)
        assert records[0]["cell_traits_per_second"] == pytest.approx(800 * 4 / 30)
        assert "8/11 chunks complete (10 dispatched)" in caplog.text
        clock[0] = 50.0
        reporter.update(1037, 11, force=True)
    assert records[-1]["completed_cells"] == 1037
    assert records[-1]["cell_traits_per_second"] == pytest.approx(1037 * 4 / 40)
    assert "recent 94.8" in caplog.text


def test_progress_persistence_error_is_nonfatal(caplog):
    def fail(record):
        raise OSError("progress disk unavailable")

    reporter = progress.LDSCProgress(1, 1, 1, callback=fail)
    reporter.update(1, 1, force=True)
    assert "Could not persist LDSC progress" in caplog.text


@pytest.mark.parametrize("traits", [1, 4])
def test_shared_progress_waits_for_results_and_counts_unpadded_tail(tmp_path, monkeypatch, traits):
    clock = [0.0]
    events, records = [], []
    monkeypatch.setattr(progress, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(shared, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    source = SimpleNamespace(
        cell_end=7,
        chunk_starts=[0, 2, 4, 6],
        ldscore_snp_positions=np.arange(3),
        close_sequential_marker_access=lambda: None,
    )

    class Batch:
        def __init__(self, width):
            self.width = width

        def wait(self):
            pass

        def to_host(self):
            events.append(("completed", self.width))
            return np.ones((traits, 2, self.width), np.float32)

    class Backend:
        name = "fake-async"

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def thread_context(self):
            return nullcontext()

        def compute(self, *, marker, ldscore):
            clock[0] += 31.0
            events.append(("dispatch", ldscore.shape[1]))
            return Batch(ldscore.shape[1])

    class Writer:
        closed = False

        def __init__(self, *args, **kwargs):
            pass

        def check_errors(self):
            pass

        def add_chunk_result(self, *args):
            pass

        def finalize(self):
            self.closed = True
            clock[0] += 1.0
            return tmp_path / "result"

        def abort(self):
            self.closed = True

    def chunks(*args):
        for start in source.chunk_starts:
            end = min(start + 2, 7)
            yield np.ones((3, end - start)), [str(i) for i in range(start, end)], start, end, None

    monkeypatch.setattr(shared, "SharedRegressionBackend", Backend)
    monkeypatch.setattr(shared, "ResultWriter", Writer)
    monkeypatch.setattr(shared, "_prefetched_chunks", chunks)
    paths = shared.run_shared_chunks(
        source,
        SimpleNamespace(),
        [{"name": str(i)} for i in range(traits)],
        None,
        tmp_path,
        7,
        progress_callback=records.append,
    )
    assert len(paths) == traits
    assert [r["completed_cells"] for r in records] == [0, 0, 2, 4, 7]
    assert records[-1]["cell_traits_per_second"] == pytest.approx(7 * traits / (124 + traits))
    assert records[-1]["completed_chunks"] == 4
    assert sum(width for kind, width in events if kind == "completed") == 7


def test_multi_gpu_wait_reports_final_progress_without_gpu():
    reports = []
    processes = [SimpleNamespace(poll=lambda: 0, wait=lambda **kwargs: 0)] * 4
    wait_workers(processes, on_progress=lambda: reports.append("done"))
    assert reports == ["done"]


def test_worker_sets_affinity_before_starting_monitor(tmp_path, monkeypatch):
    import json

    from gsMap import resource_usage
    from gsMap.spatial_ldsc.execution import multi_gpu_worker as worker

    request = {"rank": 0, "cpus": [1, 2]}
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    monkeypatch.setattr(worker.sys, "argv", ["worker", str(path)])
    events = []
    monkeypatch.setattr(worker.os, "sched_setaffinity", lambda *args: events.append("affinity"))

    def tracked(func):
        assert events == ["affinity"]
        events.append("monitor")
        return func

    monkeypatch.setattr(resource_usage, "track_resource_usage", tracked)
    monkeypatch.setattr(worker, "run_worker", lambda value: value)
    assert worker.main() == request
    assert events == ["affinity", "monitor"]


def test_multi_gpu_coordinator_aggregates_actual_worker_rows(tmp_path, monkeypatch, caplog):
    import gzip
    import json
    from pathlib import Path

    from gsMap.spatial_ldsc.execution import multi_gpu

    config = SimpleNamespace(
        ldsc_num_gpus=4,
        memmap_tmp_dir=tmp_path,
        marker_score_format="h5ad",
        marker_score_h5ad_path=tmp_path / "marker.h5ad",
        n_blocks=10,
        ldsc_trait_batch_size=4,
    )
    cells = 407
    engine = SimpleNamespace(
        config=config,
        total_cells_to_process=cells,
        chunk_size=100,
        output_dir=tmp_path,
        chunk_starts=list(range(0, cells, 100)),
        _cell_end=cells,
        spot_indices=np.arange(cells),
        spot_names_filtered=np.array([f"cell{i}" for i in range(cells)]),
        marker_score_gene_indices=np.arange(2),
        mode="weight_matrix",
        marker_score_adata=SimpleNamespace(X=np.ones((cells, 2))),
    )
    prepared = [{"name": name, "filename": name + ".csv.gz"} for name in ("IQ", "SCZ")]
    monkeypatch.setattr(
        multi_gpu, "visible_gpu_tokens", lambda count: list(map(str, range(count)))
    )
    monkeypatch.setattr(multi_gpu, "worker_cpu_sets", lambda count: [[i] for i in range(count)])

    class Worker:
        def __init__(self, command, **kwargs):
            request = json.loads(Path(command[-1]).read_text())
            staging, rank = Path(request["staging"]), request["rank"]
            metadata = json.loads((staging / "inputs.json").read_text())
            left, right = metadata["ranges"][rank]
            parts = staging / f"rank-{rank}"
            parts.mkdir()
            paths = []
            for trait in prepared:
                path = parts / trait["filename"]
                with gzip.open(path, "wt") as handle:
                    if rank == 0:
                        handle.write("spot,beta,se,z,p,neg_log10_p\n")
                    for row in range(left, right):
                        handle.write(f"cell{row},1,1,1,.1,1\n")
                paths.append(str(path))
            (staging / f"worker-{rank}.json").write_text(
                json.dumps(
                    dict(
                        range=[left, right],
                        rows=right - left,
                        paths=paths,
                        gpu_uuid=f"GPU-{rank}",
                    )
                )
            )
            Path(request["progress_path"]).write_text(
                json.dumps(
                    dict(
                        completed_cells=right - left,
                        completed_chunks=(right - left + 99) // 100,
                    )
                )
            )

        def poll(self):
            return 0

        def wait(self, **kwargs):
            return 0

    monkeypatch.setattr(multi_gpu.subprocess, "Popen", Worker)
    with caplog.at_level(logging.INFO):
        multi_gpu.run_multi_gpu(engine, np.arange(3), None, prepared)
    assert "Multi-GPU LDSC (4 GPUs) progress: 407/407 cells × 2 traits computed" in caplog.text
    assert "5/5 chunks complete" in caplog.text
    assert "including staging, worker setup and writing/merge" in caplog.text
    for trait in prepared:
        with gzip.open(tmp_path / trait["filename"], "rt") as handle:
            assert len(handle.readlines()) == cells + 1
