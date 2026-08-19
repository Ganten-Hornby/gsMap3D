"""Unit tests for the GPU double-buffering compute worker.

`LDScoreComputePool._compute_worker_pipelined` dispatches up to
``pipeline_depth`` chunks ahead and blocks lazily. The correctness properties we
guard here (without real JAX):

  * every chunk is finalized exactly once,
  * in FIFO order,
  * the trailing ``depth`` chunks are drained when the input queue runs dry
    (otherwise the monitor's chunk counter would never reach total_chunks).
"""

import queue
import threading
import time

from gsMap.spatial_ldsc.pipeline import LDScoreComputePool


def _make_worker(depth):
    """Build an LDScoreComputePool shell without running __init__/JAX setup."""
    c = object.__new__(LDScoreComputePool)
    c.pipeline_depth = depth
    c.compute_queue = queue.Queue()
    c.stop_workers = threading.Event()
    c.exception_queue = queue.Queue()
    c.has_error = threading.Event()
    c.finalized = []  # test sink
    c.inflight_high_water = 0
    c._lock = threading.Lock()

    def _dispatch(item):
        # mimic: read-failures return None and are skipped
        if not item.get("success", True):
            return None
        return item["idx"]

    def _finalize(pending):
        with c._lock:
            c.finalized.append(pending)

    c._dispatch_chunk = _dispatch
    c._finalize_chunk = _finalize
    return c


def _run(c, timeout=5.0):
    t = threading.Thread(target=c._compute_worker_pipelined, args=(0,), daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "pipelined worker did not terminate"


def test_all_chunks_finalized_in_order_with_sentinel():
    c = _make_worker(depth=2)
    n = 7
    for i in range(n):
        c.compute_queue.put({"idx": i, "success": True})
    c.compute_queue.put(None)  # close sentinel
    _run(c)
    assert c.finalized == list(range(n))  # exactly once, FIFO
    assert c.has_error.is_set() is False


def test_trailing_chunks_drained_when_queue_runs_dry():
    """No sentinel: queue empties, worker must still drain the last `depth`
    in-flight chunks (via the Empty branch) before we stop it."""
    c = _make_worker(depth=3)
    n = 5
    for i in range(n):
        c.compute_queue.put({"idx": i, "success": True})
    t = threading.Thread(target=c._compute_worker_pipelined, args=(0,), daemon=True)
    t.start()
    # Give the worker time to consume the queue and drain in-flight on Empty.
    deadline = time.time() + 5.0
    while len(c.finalized) < n and time.time() < deadline:
        time.sleep(0.02)
    c.stop_workers.set()
    t.join(2.0)
    assert not t.is_alive()
    assert c.finalized == list(range(n))


def test_read_failures_are_skipped_not_finalized():
    c = _make_worker(depth=2)
    c.compute_queue.put({"idx": 0, "success": True})
    c.compute_queue.put({"idx": 1, "success": False})  # read error -> skipped
    c.compute_queue.put({"idx": 2, "success": True})
    c.compute_queue.put(None)
    _run(c)
    assert c.finalized == [0, 2]


def test_depth_zero_still_processes_all_via_pipelined_path():
    # depth 0 through the pipelined loop degenerates to finalize-each-immediately.
    c = _make_worker(depth=0)
    for i in range(4):
        c.compute_queue.put({"idx": i, "success": True})
    c.compute_queue.put(None)
    _run(c)
    assert c.finalized == [0, 1, 2, 3]
