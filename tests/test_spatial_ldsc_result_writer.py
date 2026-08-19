import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gsMap.spatial_ldsc.result_writer import ResultWriter


def _config():
    return SimpleNamespace(
        project_name="project",
        cell_indices_range=None,
        sample_filter=None,
    )


def _add(writer, chunk_idx, rows, start):
    rows = np.asarray(rows, dtype=np.float32)
    writer.add_chunk_result(
        chunk_idx=chunk_idx,
        betas=rows,
        ses=np.ones(rows.size, dtype=np.float32),
        spot_names=pd.Index([f"cell{start + i}" for i in range(rows.size)]),
        abs_start=start,
        abs_end=start + rows.size,
    )


def test_result_writer_streams_out_of_order_chunks_atomically(tmp_path):
    writer = ResultWriter(
        config=_config(),
        trait_name="trait",
        output_dir=tmp_path,
        n_spots=5,
        total_chunks=3,
        expected_cells=5,
        output_batch_rows=2,
    )
    expected_path = tmp_path / "project_trait.csv.gz"

    _add(writer, 1, [3.0, 4.0], 2)
    assert not expected_path.exists()
    _add(writer, 0, [1.0, 2.0], 0)
    _add(writer, 2, [5.0], 4)

    output_path = writer.finalize()
    assert output_path == expected_path
    assert expected_path.is_file()
    assert not list(tmp_path.glob("*.partial.csv.gz"))

    result = pd.read_csv(output_path)
    assert result["spot"].tolist() == [f"cell{i}" for i in range(5)]
    np.testing.assert_allclose(result["beta"], np.arange(1, 6, dtype=np.float32))
    np.testing.assert_allclose(result["se"], 1.0)


def test_result_writer_rejects_duplicate_chunk_and_aborts(tmp_path):
    writer = ResultWriter(
        config=_config(),
        trait_name="trait",
        output_dir=tmp_path,
        n_spots=2,
        total_chunks=2,
        expected_cells=2,
        output_batch_rows=10,
    )
    _add(writer, 0, [1.0], 0)

    with pytest.raises(ValueError, match="Duplicate result chunk"):
        _add(writer, 0, [1.0], 0)

    writer.abort()
    assert not list(tmp_path.iterdir())


def test_result_writer_does_not_publish_incomplete_output(tmp_path):
    writer = ResultWriter(
        config=_config(),
        trait_name="trait",
        output_dir=tmp_path,
        n_spots=2,
        total_chunks=2,
        expected_cells=2,
        output_batch_rows=1,
    )
    _add(writer, 0, [1.0], 0)

    with pytest.raises(RuntimeError, match="Missing 1/2 LDSC chunks"):
        writer.finalize()

    assert not (tmp_path / "project_trait.csv.gz").exists()
    assert not list(tmp_path.glob("*.partial.csv.gz"))


def test_result_writer_backpressures_late_chunks_without_blocking_gap(tmp_path):
    writer = ResultWriter(
        config=_config(),
        trait_name="trait",
        output_dir=tmp_path,
        n_spots=4,
        total_chunks=4,
        expected_cells=4,
        output_batch_rows=10,
        max_pending_chunks=2,
    )
    _add(writer, 1, [2.0], 1)
    _add(writer, 2, [3.0], 2)

    late_finished = threading.Event()

    def add_late_chunk():
        _add(writer, 3, [4.0], 3)
        late_finished.set()

    late_thread = threading.Thread(target=add_late_chunk)
    late_thread.start()
    time.sleep(0.05)
    assert not late_finished.is_set()
    assert len(writer.pending) == writer.max_pending_chunks

    # The missing chunk bypasses the full-buffer wait and drains the gap.
    _add(writer, 0, [1.0], 0)
    late_thread.join(timeout=2)
    assert late_finished.is_set()

    result = pd.read_csv(writer.finalize())
    assert result["spot"].tolist() == ["cell0", "cell1", "cell2", "cell3"]
