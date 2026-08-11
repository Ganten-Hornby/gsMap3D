# Physical metadata audit for `Fig4to6_3DHeart.h5ad`

Audit date: 2026-08-10

- The file contains 3,095,373 cells, 238 genes, and 53 sections.
- `obsm["X_spateo"]` is the aligned 3D coordinate matrix. Its first two axes
  span approximately 5,090 by 7,743 coordinate units. The third axis is
  constant within each section and takes the synthetic sequence 150, 300, ...,
  7,950; it is therefore not a faithful physical section-center coordinate.
- `obs["z"]` is not the section axis: it varies continuously from about 0 to
  16 within every section.
- The MERFISH+ source methods report 16-um cryosections distributed round-robin
  over eight slides, with one in every eight sections imaged. The paper calls
  this 112-um inter-spacing (the unobserved gap between 16-um sections); the
  implied center-to-center interval is 128 um.
- The source reports rigid Spateo alignment of all 53 sections, with special
  partial-overlap handling for sections 21 and 22. No missing section ID occurs
  in the local file (IDs are the consecutive integers 1 through 53), although
  seven physical sections separate consecutive measured sections.

Implementation consequence: graph construction must accept an explicit
section label vector and physical 3D coordinates. For this dataset, callers
should use `graph_z_scale=128/150` to convert the synthetic third coordinate to
the same micrometre scale as XY before distance-based graph construction. The
source H5AD remains read-only.

Primary source: `MERFISH+, a large-scale, multi-omics spatial technology
resolves the molecular holograms of the 3D human developing heart`, Methods and
Figure 4 (Europe PMC article PMC12637541).
