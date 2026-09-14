"""Resource measurements must be scoped, explicit and safe on worker exit."""

import json
import logging
import subprocess
import sys
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import psutil
import pytest

from gsMap import resource_usage as usage
from gsMap.config.decorators import dataclass_typer
from gsMap.logging_config import logged_stage, stage_logging


class Process:
    def __init__(self, pid, rss, children=()):
        self.pid, self.rss, self.descendants = pid, rss, children

    def children(self, recursive):
        assert recursive
        return self.descendants

    def memory_info(self):
        if isinstance(self.rss, Exception):
            raise self.rss
        return SimpleNamespace(rss=self.rss)

    def cpu_times(self):
        return SimpleNamespace(user=2.0, system=1.0)


@pytest.fixture
def fake_procfs(monkeypatch):
    # Fake PIDs must never resolve to real /proc/1, /proc/2, etc.
    monkeypatch.setattr(usage, "_status_memory", lambda proc: {})


def test_snapshot_includes_descendants_once_and_separates_pss(monkeypatch, fake_procfs):
    child = Process(2, 200)
    root = Process(1, 100, [child, child, Process(3, 300)])
    monkeypatch.setattr(usage, "_pss_bytes", lambda proc: proc.rss // 2)
    result = usage._snapshot(root, include_pss=True)
    expected = dict(rss=100, cpu=3, tree_rss=600, tree_pss=300, processes=3, complete=True)
    expected.update(
        {f"{prefix}{key}": None for prefix in ("", "tree_") for key in usage._MEMORY_COMPONENTS}
    )
    assert result == expected


def test_vanished_child_does_not_fabricate_complete_tree_peak(monkeypatch, fake_procfs):
    root = Process(1, 100, [Process(2, psutil.NoSuchProcess(2))])
    monkeypatch.setattr(usage, "_pss_bytes", lambda proc: 50)
    result = usage._snapshot(root, include_pss=True)
    assert result["rss"] == 100
    assert result["tree_rss"] is None
    assert result["tree_pss"] is None
    assert result["tree_rss_anon"] is None
    assert not result["complete"]


def test_inaccessible_pss_does_not_become_zero(monkeypatch, fake_procfs):
    monkeypatch.setattr(usage, "_pss_bytes", lambda proc: None)
    result = usage._snapshot(Process(1, 100), include_pss=True)
    assert result["tree_rss"] == 100
    assert result["tree_pss"] is None


def test_status_reads_components_in_one_small_read(monkeypatch):
    reads = []

    def read(path):
        reads.append(str(path))
        return "Name:\ttest\nVmRSS: 100 kB\nRssAnon: 60 kB\nRssFile: 35 kB\nRssShmem: 5 kB\nVmSwap: 7 kB\n"

    monkeypatch.setattr(usage.Path, "read_text", read)
    result = usage._memory_bytes(Process(1234, 999))
    assert reads == ["/proc/1234/status"]
    assert result == dict(rss=102400, rss_anon=61440, rss_file=35840, rss_shmem=5120, swap=7168)


@pytest.mark.parametrize(
    "status",
    [
        "VmRSS: 100 kB\nRssAnon: bad kB\nRssFile: -2 kB\nRssShmem: 3 MB\n",
        "VmRSS: 100 kB\n",
    ],
)
def test_missing_malformed_components_are_not_zero(monkeypatch, status):
    monkeypatch.setattr(usage.Path, "read_text", lambda path: status)
    result = usage._memory_bytes(Process(1234, 999))
    assert result["rss"] == 102400
    assert all(result[key] is None for key in usage._MEMORY_COMPONENTS)


def test_status_read_failure_preserves_fallback_rss(monkeypatch):
    def denied(path):
        raise PermissionError("restricted procfs")

    monkeypatch.setattr(usage.Path, "read_text", denied)
    assert usage._memory_bytes(Process(1234, 456)) == dict(
        rss=456, rss_anon=None, rss_file=None, rss_shmem=None, swap=None
    )


def test_no_mixing_fallback_rss_with_partial_status(monkeypatch):
    monkeypatch.setattr(usage, "_status_memory", lambda proc: {"rss_anon": 10})
    assert usage._memory_bytes(Process(1234, 100))["rss_anon"] is None


def test_tree_missing_component_invalidates_only_that_component(monkeypatch):
    root = Process(1234, 100, [Process(1235, 200)])
    monkeypatch.setattr(
        usage,
        "_status_memory",
        lambda proc: dict(
            rss=proc.rss,
            rss_anon=proc.rss // 2,
            rss_file=proc.rss // 2,
            rss_shmem=0,
            **({"swap": 0} if proc.pid == 1234 else {}),
        ),
    )
    result = usage._snapshot(root, include_pss=False)
    assert result["tree_rss"] == 300
    assert result["tree_rss_anon"] == 150
    assert result["tree_rss_file"] == 150
    assert result["tree_rss_shmem"] == 0
    assert result["swap"] == 0
    assert result["tree_swap"] is None
    assert result["complete"]


def test_components_at_rss_peak_are_not_independent_maxima(monkeypatch, caplog):
    current = dict(
        rss=100,
        rss_anon=60,
        rss_file=40,
        rss_shmem=0,
        swap=0,
        cpu=1,
        tree_pss=None,
        processes=1,
        complete=True,
    )

    def snapshot(*args, **kwargs):
        return dict(
            current, **{f"tree_{key}": current[key] for key in ("rss", *usage._MEMORY_COMPONENTS)}
        )

    monkeypatch.setattr(usage, "_snapshot", snapshot)
    monitor = usage.ResourceMonitor()
    outer = monitor.begin()
    current.update(rss=200, rss_anon=80, rss_file=120, swap=20)
    monitor.sample()
    current.update(rss=180, rss_anon=150, rss_file=30, swap=40)
    inner = monitor.begin()
    inner_result = monitor.finish(inner)
    result = monitor.finish(outer)
    assert inner_result["memory_at_peak_rss"]["rss_anon_bytes"] == 150
    assert result["peak_rss_anon_bytes"] == 150
    assert result["peak_rss_file_bytes"] == 120
    assert result["peak_swap_bytes"] == 40
    assert result["memory_at_peak_rss"] == dict(
        rss_bytes=200, rss_anon_bytes=80, rss_file_bytes=120, rss_shmem_bytes=0, swap_bytes=20
    )
    assert result["memory_at_peak_tree_rss"] == result["memory_at_peak_rss"]
    with caplog.at_level(logging.INFO, logger="gsMap"):
        usage.log_resource_usage(result)
    assert "not additive" in caplog.text
    assert "same status read" in caplog.text
    assert "not a minimum-RAM estimate" in caplog.text


def test_nested_scopes_share_samples_and_do_not_inherit_old_peaks(monkeypatch):
    current = dict(rss=100, tree_rss=300, tree_pss=200, cpu=1.0, processes=2, complete=True)
    monkeypatch.setattr(usage, "_snapshot", lambda *args, **kwargs: dict(current))
    monitor = usage.ResourceMonitor()
    outer = monitor.begin()
    current.update(rss=150, tree_rss=500, tree_pss=400, cpu=2.0)
    monitor.sample()
    current.update(rss=120, tree_rss=150, tree_pss=100, cpu=3.0)
    inner = monitor.begin()
    current.update(cpu=4.0)
    inner_result = monitor.finish(inner)
    outer_result = monitor.finish(outer)
    assert inner_result["peak_tree_rss_bytes"] == 150
    assert outer_result["peak_tree_rss_bytes"] == 500
    assert outer_result["peak_tree_pss_bytes"] == 400
    assert outer_result["peak_rss_bytes"] == 150
    assert inner_result["self_cpu_seconds"] == 1.0
    assert outer_result["self_cpu_seconds"] == 3.0
    assert not monitor._scopes


def test_sampler_observes_mid_function_peak_and_stops(monkeypatch):
    observed = threading.Event()
    current = dict(rss=10, tree_rss=10, tree_pss=None, cpu=1.0, processes=1, complete=True)

    def snapshot(*args, **kwargs):
        if threading.current_thread().name == "gsmap-resource-monitor":
            observed.set()
        return dict(current)

    monkeypatch.setattr(usage, "_snapshot", snapshot)
    monitor = usage.ResourceMonitor(interval=0.01)
    scope = monitor.begin()
    current.update(rss=100, tree_rss=100)
    monitor.start()
    try:
        assert observed.wait(timeout=3)
        # Synchronize with the sampling thread before changing the sample.
        with monitor._lock:
            current.update(rss=10, tree_rss=10)
        result = monitor.finish(scope)
    finally:
        monitor.close()
    assert result["peak_rss_bytes"] == 100
    assert not monitor._thread.is_alive()


def test_tracking_failure_does_not_mask_original_error(monkeypatch, caplog):
    def unavailable(*args, **kwargs):
        raise psutil.AccessDenied()

    monkeypatch.setattr(usage, "_snapshot", unavailable)

    @usage.track_resource_usage
    def fail():
        raise ValueError("analysis error")

    with caplog.at_level(logging.INFO, logger="gsMap"):
        with pytest.raises(ValueError, match="analysis error"):
            fail()
    assert "main-process RSS): n/a" in caplog.text
    assert not any(t.name == "gsmap-resource-monitor" for t in threading.enumerate())


def test_real_subprocess_memory_is_included():
    monitor = usage.ResourceMonitor()
    scope = monitor.begin()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; x=bytearray(48*1024**2); print('ready', flush=True); sys.stdin.read(1)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        # Poll via communicate with a deadline while retaining the child until
        # its allocation has been sampled (the child waits on stdin).
        import select

        ready, _, _ = select.select([process.stdout], [], [], 10)
        assert ready
        assert process.stdout.readline().strip() == "ready"
        monitor.sample(force_pss=True)
        result = monitor.finish(scope)
        assert result["max_observed_processes"] >= 2
        assert result["peak_tree_rss_bytes"] > result["peak_rss_bytes"] + 40 * 1024**2
        if sys.platform == "linux":
            assert (
                result["peak_tree_rss_anon_bytes"] > result["peak_rss_anon_bytes"] + 40 * 1024**2
            )
    finally:
        process.communicate("x", timeout=10)
        monitor.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux procfs memory breakdown")
def test_real_file_mapping_and_anonymous_allocation_are_separate(tmp_path):
    path = tmp_path / "mapped.dat"
    with path.open("wb") as handle:
        handle.truncate(16 * 1024**2)
    process = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            "import mmap, sys; f=open(sys.argv[1], 'rb'); "
            "m=mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ); "
            "x=bytearray(16*1024**2); checksum=sum(m[::4096]); "
            "print('ready', flush=True); sys.stdin.read(1)",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        import select

        ready, _, _ = select.select([process.stdout], [], [], 10)
        assert ready
        assert process.stdout.readline().strip() == "ready"
        values = usage._memory_bytes(psutil.Process(process.pid))
        assert values["rss_anon"] >= 15 * 1024**2
        assert values["rss_file"] >= 15 * 1024**2
        assert values["swap"] == 0
        # Async kernel accounting is approximate; don't assert exact equality.
        assert abs(values["rss"] - sum(values[k] for k in usage._MEMORY_COMPONENTS[:3])) < 1024**2
    finally:
        process.communicate("x", timeout=10)


@pytest.mark.parametrize("interface", ["python", "cli-wrapper"])
@pytest.mark.parametrize("fail", [False, True])
def test_cli_and_python_record_one_peak_per_stage(tmp_path, interface, fail):
    @dataclass
    class Config:
        project_dir: object
        output_dir: object = None

    @logged_stage("spatial_ldsc")
    def python_stage(config):
        assert sum(t.name == "gsmap-resource-monitor" for t in threading.enumerate()) == 1
        if fail:
            raise ValueError("stage error")
        return 42

    # Build the same wrapper as the CLI, without initializing a device.
    def spatial_ldsc(config: Config):
        return python_stage(config)

    cli_stage = dataclass_typer(spatial_ldsc)

    def run():
        return (
            python_stage(Config(tmp_path))
            if interface == "python"
            else cli_stage(project_dir=tmp_path, output_dir=tmp_path)
        )

    if fail:
        with pytest.raises(ValueError, match="stage error"):
            run()
    else:
        assert run() == 42
    latest = tmp_path / "logs/latest"
    text = (latest / "03_spatial_ldsc.log").read_text()
    assert text.count("Peak memory usage") == 1
    manifest = json.loads((latest / "run_manifest.json").read_text())
    assert len(manifest["stages"]) == 1
    resource = manifest["stages"][0]["resource_usage"]
    assert resource["peak_rss_bytes"] > 0
    assert resource["samples"] >= 2
    assert resource["wall_seconds"] > 0
    if sys.platform == "linux":
        assert resource["peak_rss_anon_bytes"] > 0
        assert resource["peak_rss_file_bytes"] > 0
        assert resource["memory_at_peak_rss"]["rss_bytes"] == resource["peak_rss_bytes"]
    assert not any(t.name == "gsmap-resource-monitor" for t in threading.enumerate())


def test_quick_mode_tracks_nested_steps_with_one_sampler(tmp_path):
    with stage_logging(tmp_path, "quick_mode") as session:
        with stage_logging(tmp_path, "spatial_ldsc"):
            with stage_logging(tmp_path, "spatial_ldsc"):
                assert sum(t.name == "gsmap-resource-monitor" for t in threading.enumerate()) == 1
        with stage_logging(tmp_path, "report"):
            pass
    manifest = json.loads(session.manifest_path.read_text())
    assert len(manifest["stages"]) == 3
    assert all(s["resource_usage"]["peak_rss_bytes"] > 0 for s in manifest["stages"])
    assert (session.run_dir / "03_spatial_ldsc.log").read_text().count("Peak memory usage") == 1


def test_ldscore_pipeline_python_entry_tracks_resources(tmp_path, monkeypatch):
    from gsMap.ldscore.pipeline import LDScorePipeline

    pipeline = LDScorePipeline.__new__(LDScorePipeline)
    pipeline.config = SimpleNamespace(output_dir=tmp_path, annot_file="annotation")
    monkeypatch.setattr(pipeline, "_run_with_annotation", lambda: None)
    pipeline.run()
    text = (tmp_path / "logs/latest/ldscore_weight_matrix.log").read_text()
    assert text.count("Peak memory usage") == 1


def test_format_sumstats_python_failure_still_tracks_resources(tmp_path):
    from gsMap.format_sumstats import gwas_format

    config = SimpleNamespace(sumstats=tmp_path / "missing.txt", out=tmp_path / "formatted")
    with pytest.raises(FileNotFoundError):
        gwas_format(config)
    text = (tmp_path / "logs/latest/format_sumstats.log").read_text()
    assert text.count("Peak memory usage") == 1
