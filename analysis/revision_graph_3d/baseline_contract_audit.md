# Corrected 2D graph contract audit

Audit date: 2026-08-10

## Confirmed implementation contract

- `FindLatentModelConfig.n_neighbors` defaults to 10 and is documented as the
  number of external neighbors.
- `build_spatial_graph` queries external neighbors explicitly: it excludes the
  focal node before selecting the nearest K nodes. For a section with fewer
  than K other spots, the effective K is `n_section_spots - 1`.
- The directed pre-symmetrization selection has exactly effective-K outgoing
  edges per eligible spot and contains no self edges.
- The production graph is the union of each focal spot's KNN choices and their
  reverses. Consequently it contains both directed representations of every
  external relation. Reciprocal selections are coalesced, so final external
  degree is at least effective-K and can exceed K when a spot is selected by a
  non-reciprocal neighbor.
- `build_spatial_graph` returns no self edges. `GCN.forward` calls `gcn_norm`
  with `add_self_loops=True`, adding the message-passing self-loop separately.
- Stable observation IDs provide deterministic tie-breaking. The graph and GCN
  propagation tests confirm invariance/equivariance under observation-row
  permutation, including tied and duplicated coordinates.

## Verification

Command:

```text
.venv/bin/pytest -q tests/test_spatial_graph.py
```

Result: 10 passed in 74.37 seconds. The 42 warnings were upstream deprecation
warnings from Torch and PyTorch Geometric; there were no test failures.

The focused tests cover exact external-K selection before symmetrization,
absence of self edges, explicit bidirectionality, permutation invariance,
duplicate-coordinate tie handling, sections smaller than K, separate GCN
self-loop normalization, and propagation equivariance. The implementation and
docstring explicitly record the reciprocal-degree consequence after the KNN
union is symmetrized.

## Output provenance boundary

The correction was introduced by commit `93e47ec` on 2026-08-08 00:11 +08:00.
The only `find_latent_metadata.yaml` currently under `analysis/` is:

`analysis/eqtl_embryo_e16/work/E16.5_E1S1_eqtl/find_latent_representations/find_latent_metadata.yaml`

It and its latent/training H5AD outputs were written on 2026-06-12. Its
metadata records `n_neighbors: 10` but has no `graph_mode`, `graph_radius`, or
`graph_sigma`, showing that it predates the corrected graph-mode contract.
Therefore these existing analysis outputs are historical pre-correction
outputs. No post-correction latent output was found under `analysis/` as of the
audit date; none may be cited as a corrected rerun yet.
