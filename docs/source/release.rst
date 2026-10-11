Release notes
=============

v0.1.4a2
--------

Install this alpha release explicitly with ``pip install gsMap3D==0.1.4a2``.

* Explicit ``expression_data_type=noncounts`` support for rank-only marker
  scoring with an existing cell embedding. This mode skips ``find_latent``,
  requires ``no_expression_fraction=True``, and rejects count-based QC options.
  Counts remain the default; automatic expression-type detection is not yet
  included.
* A GWAS-independent ``DataPreparationConfig`` for configuring the latent
  representation and marker-score stages.
* Clearer 2D/3D coordinate validation and support for single-volume thick-tissue
  inputs, with cross-section matching disabled for a single 3D input.
* Numeric pandas DataFrame spatial coordinates are accepted.
* Updated report headers, spatial cell banners, plotting guidance, tutorial
  resources, documentation navigation, and resource-mirror configuration.

This pre-release does not update stable Docker tags.

v0.1.4a1
--------

Alpha release of the revised 2D/3D workflow. Install this preview explicitly
with ``pip install gsMap3D==0.1.4a1``.

* Shared multi-trait spatial LDSC, optimized CPU/GPU regression, and single-node
  multi-GPU execution.
* Faster GWAS preparation and per-stage resource and throughput logging.
* Improved interactive reports and automatic SpatialVista export for 3D data.
* Expression-based cell filtering before latent modeling and HVG provenance.

This pre-release does not update stable Docker tags.

Current features and improvements
---------------------------------

* Ordinal matching defaults to two sections on each side.
* Quick mode exposes the exact external-neighbor count, radius and bandwidth.
* Executable 2D/3D tutorials use paired V50 LD references and report previews.
* The E11.5 example runs ordinal ±8-section matching in one quick-mode call.
* Cauchy aggregation aligns reordered LDSC rows by unique spot ID rather than
  assuming the canonical AnnData order. Duplicate or mismatched IDs are rejected.
* LD calculation accepts finite, ordered cM maps with a negative starting origin;
  constant, non-finite and unsorted maps remain invalid.
* Resource bundles include SHA-256 manifests and matching regression weights.

See :doc:`advanced_usage` for configuration examples and :doc:`downloads` for
resource availability and shared setup instructions.

v0.1.0
------

First public release
