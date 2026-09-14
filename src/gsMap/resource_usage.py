"""Sample host memory without initializing an accelerator or querying Slurm.

RSS includes resident file mappings. Summed process-tree RSS counts shared pages
more than once; Linux PSS apportions them between processes. Neither is VRAM.
All peaks are sampled within a scope, not the process-lifetime high-water mark.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from functools import wraps
from pathlib import Path

import psutil

logger = logging.getLogger("gsMap")

_MEMORY_COMPONENTS = ("rss_anon", "rss_file", "rss_shmem", "swap")
_STATUS_FIELDS = {
    "VmRSS": "rss",
    "RssAnon": "rss_anon",
    "RssFile": "rss_file",
    "RssShmem": "rss_shmem",
    "VmSwap": "swap",
}


def _status_memory(process):
    """Read cheap Linux counters together; missing fields stay unavailable."""
    try:
        text = Path(f"/proc/{process.pid}/status").read_text()
    except OSError:
        return {}
    result = {}
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if not separator or name not in _STATUS_FIELDS:
            continue
        fields = value.split()
        try:
            if len(fields) == 2 and fields[1] == "kB" and int(fields[0]) >= 0:
                result[_STATUS_FIELDS[name]] = int(fields[0]) * 1024
        except ValueError:
            continue
    return result


def _memory_bytes(process):
    memory = _status_memory(process)
    if "rss" not in memory:
        # Keep RSS on non-Linux hosts or when procfs fields are unavailable.
        # Do not combine an unrelated fallback RSS sample with partial fields.
        memory = {"rss": process.memory_info().rss}
    return {key: memory.get(key) for key in ("rss", *_MEMORY_COMPONENTS)}


def _pss_bytes(process):
    """Use Linux's small rollup, not a scan of every memory mapping."""
    try:
        for line in Path(f"/proc/{process.pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def _snapshot(process, *, include_pss):
    rss = cpu = None
    components = dict.fromkeys(_MEMORY_COMPONENTS)
    tree_components = dict.fromkeys(_MEMORY_COMPONENTS, 0)
    tree_rss = 0
    tree_pss = 0 if include_pss else None
    complete = True
    try:
        children = process.children(recursive=True)
    except (OSError, psutil.Error):
        children, complete = [], False
    # Defensive de-duplication also prevents counting a descendant twice.
    processes = {p.pid: p for p in [process, *children]}
    for pid, proc in processes.items():
        try:
            memory = _memory_bytes(proc)
            tree_rss += memory["rss"]
            for key in _MEMORY_COMPONENTS:
                value = memory[key]
                tree_components[key] = (
                    None
                    if value is None or tree_components[key] is None
                    else tree_components[key] + value
                )
            if pid == process.pid:
                rss = memory["rss"]
                components = {key: memory[key] for key in _MEMORY_COMPONENTS}
                times = proc.cpu_times()
                cpu = times.user + times.system
            if include_pss:
                pss = _pss_bytes(proc)
                tree_pss = None if pss is None or tree_pss is None else tree_pss + pss
        except (OSError, psutil.Error):
            # A worker can exit between discovery and sampling. Do not label a
            # partially observed tree as a complete measurement.
            complete = False
    return dict(
        rss=rss,
        cpu=cpu,
        tree_rss=tree_rss if complete else None,
        tree_pss=tree_pss if complete else None,
        processes=len(processes),
        complete=complete,
        **components,
        **{f"tree_{key}": value if complete else None for key, value in tree_components.items()},
    )


class ResourceMonitor:
    """One sampler shared by nested stage scopes in a CLI or Python run."""

    def __init__(self, interval=0.5, pss_interval=5.0):
        if interval <= 0 or pss_interval <= 0:
            raise ValueError("Resource sampling intervals must be positive")
        self.interval = interval
        self.pss_interval = pss_interval
        self.process = psutil.Process(os.getpid())
        self._lock = threading.RLock()
        self._scopes = []
        self._stop = threading.Event()
        self._thread = None
        self._last_pss = float("-inf")

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="gsmap-resource-monitor", daemon=True
            )
            self._thread.start()

    def _run(self):
        while not self._stop.wait(self.interval):
            self.sample()

    def _read(self, force_pss=False):
        now = time.monotonic()
        include_pss = force_pss or now - self._last_pss >= self.pss_interval
        try:
            snapshot = _snapshot(self.process, include_pss=include_pss)
        except Exception as error:  # noqa: BLE001 - telemetry must not fail an analysis
            logger.debug("Resource sample unavailable: %s", error)
            snapshot = dict(
                rss=None, cpu=None, tree_rss=None, tree_pss=None, processes=0, complete=False
            )
        if include_pss:
            self._last_pss = now
        return snapshot

    @staticmethod
    def _update(scope, snapshot):
        scope["samples"] += 1
        scope["incomplete_tree_samples"] += not snapshot["complete"]
        scope["max_observed_processes"] = max(
            scope["max_observed_processes"], snapshot["processes"]
        )
        for prefix in ("", "tree_"):
            value = snapshot.get(f"{prefix}rss")
            peak = scope[f"peak_{prefix}rss_bytes"]
            if value is not None and (peak is None or value > peak):
                # Preserve the components of this sample, not independent maxima.
                scope[f"memory_at_peak_{prefix}rss"] = {
                    f"{key}_bytes": snapshot.get(f"{prefix}{key}")
                    for key in ("rss", *_MEMORY_COMPONENTS)
                }
        for key in (
            "rss",
            "tree_rss",
            "tree_pss",
            *_MEMORY_COMPONENTS,
            *(f"tree_{key}" for key in _MEMORY_COMPONENTS),
        ):
            value = snapshot.get(key)
            if value is not None:
                name = f"peak_{key}_bytes"
                scope[name] = value if scope[name] is None else max(scope[name], value)
        if snapshot["cpu"] is not None:
            if scope["_cpu_start"] is None:
                scope["_cpu_start"] = snapshot["cpu"]
            scope["_cpu_end"] = snapshot["cpu"]

    def sample(self, *, force_pss=False):
        with self._lock:
            if self._scopes:
                snapshot = self._read(force_pss)
                for scope in self._scopes:
                    self._update(scope, snapshot)

    def begin(self):
        with self._lock:
            scope = dict(
                peak_rss_bytes=None,
                peak_tree_rss_bytes=None,
                peak_tree_pss_bytes=None,
                **{
                    f"peak_{prefix}{key}_bytes": None
                    for prefix in ("", "tree_")
                    for key in _MEMORY_COMPONENTS
                },
                memory_at_peak_rss=None,
                memory_at_peak_tree_rss=None,
                samples=0,
                incomplete_tree_samples=0,
                max_observed_processes=0,
                sample_interval_seconds=self.interval,
                pss_interval_seconds=self.pss_interval,
                _started=time.monotonic(),
                _cpu_start=None,
                _cpu_end=None,
            )
            self._scopes.append(scope)
            self.sample(force_pss=True)
            return scope

    def finish(self, scope):
        with self._lock:
            self.sample(force_pss=True)
            self._scopes = [active for active in self._scopes if active is not scope]
            result = {key: value for key, value in scope.items() if not key.startswith("_")}
            result["wall_seconds"] = max(0.0, time.monotonic() - scope["_started"])
            result["self_cpu_seconds"] = (
                None
                if scope["_cpu_start"] is None
                else max(0.0, scope["_cpu_end"] - scope["_cpu_start"])
            )
            result["self_cpu_utilization_percent"] = (
                None
                if result["self_cpu_seconds"] is None
                else 100.0 * result["self_cpu_seconds"] / max(result["wall_seconds"], 1e-9)
            )
            return result

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()


def log_resource_usage(result, *, label="function"):
    def gib(value):
        return "n/a" if value is None else f"{value / 1024**3:.2f} GiB"

    logger.info("Resource usage summary | %s", label)
    logger.info(
        "  • Peak memory usage (sampled main-process RSS): %s",
        gib(result["peak_rss_bytes"]),
    )
    logger.info(
        "  • Peak process-tree RSS (sum; shared pages counted per process): %s",
        gib(result["peak_tree_rss_bytes"]),
    )
    logger.info(
        "  • Peak process-tree PSS (proportional shared pages, where available): %s",
        gib(result["peak_tree_pss_bytes"]),
    )
    logger.info(
        "  • Main-process component peaks (not additive): RssAnon=%s; RssFile=%s; "
        "RssShmem=%s; VmSwap=%s",
        *(gib(result.get(f"peak_{key}_bytes")) for key in _MEMORY_COMPONENTS),
    )
    if result["max_observed_processes"] > 1:
        logger.info(
            "  • Process-tree component peaks (sums may double-count shared pages): "
            "RssAnon=%s; RssFile=%s; RssShmem=%s; VmSwap=%s",
            *(gib(result.get(f"peak_tree_{key}_bytes")) for key in _MEMORY_COMPONENTS),
        )
    at_peak = result.get("memory_at_peak_rss") or {}
    logger.info(
        "  • At main-process RSS peak (same status read): RssAnon=%s; RssFile=%s; "
        "RssShmem=%s; VmSwap=%s",
        *(gib(at_peak.get(f"{key}_bytes")) for key in _MEMORY_COMPONENTS),
    )
    cpu = result["self_cpu_seconds"]
    utilization = result["self_cpu_utilization_percent"]
    logger.info(
        "  • Wall time: %.2fs; main-process CPU time: %s; CPU utilization: %s; "
        "max observed processes: %d",
        result["wall_seconds"],
        "n/a" if cpu is None else f"{cpu:.2f}s",
        "n/a" if utilization is None else f"{utilization:.1f}% (100% = one core)",
        result["max_observed_processes"],
    )
    logger.info(
        "  • Host RAM only, not GPU VRAM; RSS sampled every %.1fs, PSS every %.1fs "
        "and at stage boundaries; short peaks may be missed; incomplete tree samples: %d",
        result["sample_interval_seconds"],
        result["pss_interval_seconds"],
        result["incomplete_tree_samples"],
    )
    logger.info(
        "  • Linux RSS counters are approximate; RssAnon excludes file mappings and "
        "shared memory, includes allocator-retained memory, and is not a minimum-RAM estimate."
    )


def track_resource_usage(func):
    """Explicit tracking for custom Python functions (public stages track automatically)."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = ResourceMonitor()
        scope = monitor.begin()
        monitor.start()
        try:
            return func(*args, **kwargs)
        finally:
            try:
                log_resource_usage(monitor.finish(scope), label=func.__name__)
            finally:
                monitor.close()

    return wrapper
