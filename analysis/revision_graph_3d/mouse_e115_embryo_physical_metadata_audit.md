# Mouse E11.5 embryo physical-metadata audit

- Source: `/storage/yangjianLab/lilounan/data/3dst/Mouse_E11.5_embryo.h5ad`
- Observation count: 6,993,667.
- Section identity: `obs["slice_id"]`, containing 100 labels.
- Graph coordinates: `obsm["z_correction"]`, a 6,993,667 × 3 `float64` matrix.
- Coordinate ranges: x = 580.2225–1,851.6861, y = 38.6099–1,625.5830,
  z = 305–800 source coordinate units.
- Section ordering uses the median corrected z coordinate for each `slice_id`.
  The 100 labels occupy 79 distinct median-z positions; 21 positions are shared
  by two labels. Sixty-eight labels contain spots at two corrected z values.
- The 78 positive gaps between distinct median-z positions comprise 64 gaps of
  5 units, 10 of 10 units, two of 15 units, one of 20 units, and one of 25
  units. The modal distinct center-to-center spacing is therefore 5 source
  coordinate units, with larger gaps retained rather than imputed.
- The H5AD `uns` group is empty. Neither coordinate units nor physical section
  thickness are documented in the supplied file. Distances must therefore be
  reported in source coordinate units; no micrometre or millimetre conversion
  is assumed. Missing tissue cannot be distinguished from acquisition gaps using
  the supplied metadata alone and remains unresolved.
- The revision run uses `graph_z_scale = 1.0`; no additional z recalibration is
  applied beyond the supplied `z_correction` coordinates.
