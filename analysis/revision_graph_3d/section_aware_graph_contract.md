# Section-aware physical 3D graph contract

For each focal spot:

1. Select up to K external nearest neighbors within the same section.
2. Define the spot's local radius as the 3D physical distance to the farthest
   selected within-section neighbor.
3. Search the immediately preceding and following sections, ordered by median
   physical z, and connect every cross-section spot within that radius.
4. If the combined radius search across those neighboring sections returns no
   spot, consider up to `max(1, K // 2)` nearest spots from each available
   neighboring section, but keep only candidates within twice the focal
   within-section radius. A sufficiently large physical gap therefore yields
   no cross-section edge.
5. Materialize both directed representations of every selected external edge,
   coalesce duplicates, and leave self-loops for GCN normalization.

The graph requires an explicit observation section column and coordinates in a
common calibrated physical XYZ scale. It does not reuse the later homogeneous
neighbor matching or expression-averaging operation.

CLI configuration:

```text
--graph-mode section_aware_3d
--graph-section-key <obs-column>
--spatial-key <physical-xyz-obsm-key>
--graph-z-scale <stored-z-to-physical-xy multiplier>
--n-neighbors 10
```

All aligned sections must currently reside in a single H5AD so training and
inference construct the same unified graph. The builder requires exactly three
XYZ columns and rejects higher-dimensional representations.

Focused tests cover the radius rule, the distance-ceiling fallback, zero edges
across large physical gaps, explicit bidirectionality, absence of self edges,
stable tie handling, row-order invariance, and CLI exposure.
