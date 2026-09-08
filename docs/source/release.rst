Release notes
=============

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
