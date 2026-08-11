# Mouse E11.5 heart annotation audit

## Input

- Source: `/storage/yangjianLab/lilounan/data/3dst/Mouse_E11.5_heart.h5ad`
- Shape: 98,966 observations × 19,746 genes
- Observation IDs: globally unique within the file
- Sections: 64 non-missing `slices` categories, with 61–3,042 observations per section

## Physical metadata audit

- `obsm["aligned_spatial_3D_mm"]` is the exact coordinate matrix used by the
  3D arm. The key names its unit as millimetres; observed XYZ ranges are
  approximately 0.0009–0.6960, 0–0.6517, and 0–0.7600 mm.
- Z is constant within each section. Ordered section centers span 0–0.76 mm.
  Of the 63 successive center-to-center intervals, 56 are 0.01 mm, four are
  0.02 mm, and one each is 0.03, 0.04, and 0.05 mm. The analysis preserves
  these nonuniform stored distances.
- All labels `slices_1` through `slices_64` are present, but the H5AD contains
  no acquisition metadata or `uns` entries that distinguish skipped physical
  sections from enlarged inter-section gaps. Missing-tissue status is likewise
  not encoded and is therefore unknown from the supplied artifact.
- The Spateo mouse-heart tutorial identifies this exact 98,966-observation
  object and its coordinate fields as processed MOSTA3D data. However, neither
  that processed H5AD nor its embedded metadata records section thickness, so
  thickness is recorded as unknown rather than inferred from another dataset.
- No z recalibration was applied: `graph_z_scale=1.0`, because all three axes
  in `aligned_spatial_3D_mm` are already expressed in the same stored unit.

Sources: the supplied H5AD; the Spateo mouse-heart morphogenesis tutorial; and
the MOSTA3D preprint (doi:10.1101/2024.08.17.608366).

## Candidate annotations

| Field | Non-missing | Categories | Assessment |
|---|---:|---:|---|
| `ctype_user` | 98,966 | 35 | Cell-type/lineage metadata; technically valid but extremely imbalanced |
| `cml` | 98,966 | 2 | Exact coarse partition of `ctype_user` into cardiac muscle lineage versus other |
| `heart_anno` | 98,966 | 1 | Constant (`Heart`); not informative for training or evaluation |
| `heart_regions` | 98,966 | 5 | User-selected two-stage training target; post-training fit is not independent validation |
| `stage` | 98,966 | 1 | Constant (`E11.5`); metadata only |

`ctype_user` contains 93,809 `Cardiac muscle lineages` observations (94.79%) and
5,157 observations across the other 34 categories. The `cml` labels agree exactly
with this split. Several rare `ctype_user` categories contain fewer than 10
observations, so this label is not suitable for an unweighted two-stage
classification objective without a separate imbalance strategy.

The `heart_regions` counts are:

| Region | Observations |
|---|---:|
| Right ventricle | 32,652 |
| Left ventricle | 31,453 |
| Right atrium | 14,967 |
| Left atrium | 11,554 |
| Outflow tract | 8,340 |

## Decision for the matched 2D-versus-3D embedding run

- Per the user's explicit direction, use `heart_regions` as the training
  annotation with `--two-stage` so the five region labels enter the second-stage
  classification objective.
- Use `--no-sampling`, so all 98,966 observations are used in both arms.
- Keep all model, data, seed, and training settings identical between arms.
- Treat any post-training `heart_regions` separation or purity as training-label
  fit, not independent spatial-domain validation. Use `ctype_user`, physical-edge
  continuity, and 2D–3D neighborhood overlap as non-target comparison axes.
