# Physical metadata audit for Mouse E9.5

Audit date: 2026-08-11

- Source: `/storage/yangjianLab/lilounan/data/3dst/E9.5/Mouse_E9.5_embryo.h5ad`
- Coordinates: `obsm["3d_align_spatial"]`; section labels are derived from its third coordinate and stored as `obs["section_id"]` in the prepared input.
- The dataset contains 70 distinct section-center coordinates spanning 0 to 1,760 source coordinate units.
- Of the 69 successive center-to-center gaps, 61 are 20 units, one is 40 units, five are 60 units, and two are 100 units. The modal center spacing is therefore 20 source coordinate units.
- The source H5AD has an empty `uns` group and no observation column documenting coordinate units or section thickness. Physical units and section thickness are therefore **not documented in the supplied file**; report distances as source coordinate units rather than assuming micrometres.
- Larger numerical z gaps are preserved, but the file does not distinguish deliberately unmeasured sections from missing tissue. Do not infer tissue continuity across them.
- The section-aware run used `graph_z_scale: 1.0`, so no z recalibration was applied.

The prepared-input provenance is recorded in
`analysis/revision_graph_3d/inputs/Mouse_E9.5_embryo_sectioned.provenance.json`.
