"""Bounded trait preparation: no executor.map queue of all GWAS files."""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from numbers import Integral

from gsMap.spatial_ldsc.backends.cpu_sparse import available_cpu_count


def resolve_gwas_workers(requested, count):
    if (
        isinstance(requested, bool)
        or not isinstance(requested, Integral)
        or not 1 <= requested <= 16
    ):
        raise ValueError("ldsc_gwas_workers must be an integer between 1 and 16")
    return max(1, min(requested, count, available_cpu_count()))


def bounded_trait_map(function, items, workers, *, ordered=True):
    """Yield compact prepared arrays, retaining <= workers futures.

    The function must discard raw GWAS tables before returning. At most workers
    files are parsing/preparing/awaiting consumption, independent of trait count.
    Closing the iterator cancels queued work and joins running readers.
    Set ordered=False to refill workers immediately on any file's completion.
    """
    if workers == 1:
        for item in items:
            yield function(item)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ldsc-gwas-reader") as pool:
        pending = deque()
        try:
            for _ in range(workers):
                item = next(iterator, None)
                if item is None:
                    break
                pending.append(pool.submit(function, item))
            while pending:
                if ordered:
                    future = pending.popleft()
                else:
                    ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                    future = next(iter(ready))
                    pending.remove(future)
                    del ready
                yield future.result()
                del future
                item = next(iterator, None)
                if item is not None:
                    pending.append(pool.submit(function, item))
        finally:
            for future in pending:
                future.cancel()
