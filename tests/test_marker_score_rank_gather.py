import threading

import numpy as np
import pytest

import gsMap.latent2gene.marker_scores as marker_score_module
from gsMap.latent2gene.marker_scores import ParallelMarkerScoreComputer


class _CountingLock:
    def __init__(self):
        self._lock = threading.Lock()
        self.entries = 0

    def __enter__(self):
        self._lock.acquire()
        self.entries += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._lock.release()


def _computer_without_workers():
    computer = object.__new__(ParallelMarkerScoreComputer)
    computer.rank_gather_lock = _CountingLock()
    return computer


def test_rank_gather_reconstructs_exact_rows_under_lock():
    computer = _computer_without_workers()
    rank_data = np.arange(12, dtype=np.float16).reshape(3, 4)
    inverse = np.array([2, 0, 1, 2], dtype=np.int32)

    observed = np.asarray(computer._reconstruct_batch_ranks(rank_data, inverse))

    assert np.array_equal(observed, rank_data[inverse])
    assert computer.rank_gather_lock.entries == 1


def test_rank_gather_blocks_before_releasing_lock(monkeypatch):
    events = []

    class RecordingLock:
        held = False

        def __enter__(self):
            self.held = True
            events.append("enter")

        def __exit__(self, exc_type, exc_value, traceback):
            events.append("exit")
            self.held = False

    lock = RecordingLock()
    computer = object.__new__(ParallelMarkerScoreComputer)
    computer.rank_gather_lock = lock

    class FakeBatchRanks:
        def block_until_ready(self):
            assert lock.held
            events.append("block")

    class FakeRankData:
        def __getitem__(self, indices):
            del indices
            assert lock.held
            events.append("gather")
            return FakeBatchRanks()

    def fake_asarray(value):
        assert lock.held
        if np.asarray(value).ndim == 2:
            events.append("rank_transfer")
            return FakeRankData()
        events.append("index_transfer")
        return object()

    monkeypatch.setattr(marker_score_module.jnp, "asarray", fake_asarray)
    computer._reconstruct_batch_ranks(
        np.arange(12, dtype=np.float16).reshape(3, 4),
        np.array([2, 0, 1, 2], dtype=np.int32),
    )

    assert events == [
        "enter",
        "rank_transfer",
        "index_transfer",
        "gather",
        "block",
        "exit",
    ]


@pytest.mark.parametrize("inverse", [[-1, 0], [0, 3]])
def test_rank_gather_rejects_indices_outside_slab(inverse):
    computer = _computer_without_workers()
    rank_data = np.arange(12, dtype=np.float16).reshape(3, 4)

    with pytest.raises(IndexError, match="outside slab bounds"):
        computer._reconstruct_batch_ranks(rank_data, np.asarray(inverse, dtype=np.int32))


@pytest.mark.parametrize(
    ("rank_data", "inverse", "match"),
    [
        (np.zeros(3, dtype=np.float16), np.array([0]), "Rank slab must be 2D"),
        (np.zeros((3, 2), dtype=np.float16), np.empty(0, dtype=np.int32), "non-empty vector"),
    ],
)
def test_rank_gather_validates_payload_shape(rank_data, inverse, match):
    computer = _computer_without_workers()

    with pytest.raises(ValueError, match=match):
        computer._reconstruct_batch_ranks(rank_data, inverse)
