# gsMap3D 3D-Graph Revision Rules

## Project Configuration

- `PROJECT_ROOT`: `/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning`
- `WORKFLOW_DB`: `/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/REBUTTAL_WORKFLOW.sqlite`
- `REBUTTAL_SOURCE`: `/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/rebuttal/rebuttal.qmd`
- Source review plan: `/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/comments_response_plan.docx`
- Source review plan SHA-256: `b9660d97ba8fc3777c5cfa2655bd9af5270a5d85c0780c239e94febe3671edf8`
- Working branch: `3d-graph-learning`
- Authoritative revision worktree: `/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning`. Keep all revision code, analyses, logs, and artifacts in this worktree; do not create ad hoc alternative revision worktrees, use opaque scratch aliases such as `meta_lcl`, or place revision edits in the main worktree.
- The manuscript/SI project root, render command, local deliverable, and sync target are not configured yet. Do not apply manuscript edits, render, or sync until those values are resolved.

`REBUTTAL_WORKFLOW.sqlite` is the only task and dependency ledger. Do not create or maintain a Markdown task queue.

## Worktree, Environment, and Slurm Rules

- Use the existing `${PROJECT_ROOT}/.venv`; do not create another virtual environment for revision work or for CUDA troubleshooting.
- Keep PyTorch, DGL, and CUDA dependencies compatible with CUDA 12. Do not introduce a new CUDA major-version dependency unless a concrete package requires it, the requirement is verified on a GPU node, and the user explicitly approves the environment change.
- A login node is not evidence for CUDA or DGL GPU compatibility. Run CUDA health checks, GPU imports, and GPU smoke tests inside an allocated GPU node.
- Submit computational workloads through Slurm with callback coverage; do not leave a raw `sbatch` job without a registered callback. Use scheduler/resource-database evidence for CPU and memory requests.
- Omit a Codex-imposed `--time` limit unless the user explicitly asks for one or the scheduler requires it. Cluster or QoS defaults may still apply.
- For repeated OOM failures, continue without asking the user to reconfirm the same recovery policy: raise memory using accounting evidence or implement a memory-bounded code fix. If memory use follows the allocation ceiling or a second enlarged request also OOMs, prefer a code fix over further blind escalation.

## Focused Response Scope

The revision project covers the four core graph-learning comments and two directly supporting validation comments below. Status, priority, dependencies, and artifacts belong in the workflow database, not in this file.

| Response ID | Reviewer location | Scope |
|---|---|---|
| `R2C1` | Reviewer 2, Major 1 | Unified physical 3D graph, cross-section edges, measured z, section thickness/gaps, and missing-section robustness |
| `R2C2` | Reviewer 2, Major 2 | Definition of K, external neighbors versus self-loop, physical neighborhood scale, local density, and graph sensitivity analyses |
| `R2C3` | Reviewer 2, Major 3 | Component attribution and ablations separating 3D organization from pooling, matching, and embedding choices |
| `R2C6` | Reviewer 2, Major 6 | Bidirectional-edge bug, row-order invariance, and before/after impact audit |
| `R3C2` | Reviewer 3, limitation (b) | Calibration and false-positive checks for cross-section neighbor averaging |
| `R3C5` | Reviewer 3, limitation (e) | Orthogonal validation with alternative 3D reconstruction/alignment methods |

Comments outside this scope are not to be added unless the user explicitly broadens the project.

## Authored Comment and Response Workspace

- Canonical ordered source: `rebuttal/rebuttal.qmd`
- Reviewer comment excerpts: `rebuttal/comments/<response-id>.md`
- Author response fragments: `rebuttal/responses/<response-id>.qmd`
- Each response fragment must contain exactly one `.author-response` block whose `response-id` matches its filename and SQLite response ID.
- Comment excerpts are source records, not task state. Reconcile any correction against `comments_response_plan.docx` and its recorded SHA-256.
- Response fragments remain empty until the corresponding analysis task is complete and the automatically enqueued response task has been started.
- Do not place task checkboxes, priorities, statuses, or dependencies in these files; those remain exclusively in `REBUTTAL_WORKFLOW.sqlite`.

## Scientific Contract for This Revision

- The default within-section graph means **10 external spatial neighbors per spot**. A self-loop is a separate message-passing operation and must not be counted as an external neighbor.
- Sections are already aligned. Do not present optimal transport as necessary for alignment or as a mechanism that solves a large physical section gap.
- The working cross-section proposal is deliberately simple: use measured physical 3D distance and choose cross-section neighbors so that their effective sampling density is matched to the within-section 10-neighbor graph. The analysis must make the density-matching rule explicit and test it rather than silently imposing a fixed cross-section neighbor count.
- A large z gap may legitimately yield few or no direct cross-section edges at the within-section physical scale. Do not force a fixed number of cross-section neighbors merely because two sections are consecutive in slice order.
- Any discussed `k/2` nearest-neighbor fallback must still obey an explicit physical-distance or gap ceiling. Slice adjacency alone is never sufficient justification for an edge.
- Distinguish graph edges used for GNN message passing from later homogeneous-spot matching or expression averaging. Report each operation separately.
- Record section thickness, center-to-center spacing, missing tissue, coordinate units, and any z recalibration for every dataset used in the revision.

## Graph Implementation and Memory-Safety Rules

- The discoverable section-aware implementation is `build_section_aware_3d_graph` in `src/gsMap/find_latent/gnn/gcn.py`; `build_spatial_graph` delegates to it when `mode == "section_aware_3d"`. Do not hide a second implementation in an analysis script or another worktree.
- Preserve propagation direction explicitly and cover unweighted, weighted, and legacy-normalized paths with numerical parity tests against an independent edge-message reference.
- GCN propagation must remain sparse and memory-bounded. Do not materialize an edge-by-feature tensor or any dense equivalent proportional to `n_edges × feature_width`; use sparse aggregation such as indexed accumulation or sparse matrix multiplication.
- A full exact-shape representative real-data run is required after a propagation-memory change. Synthetic parity and allocation-regression tests are necessary but do not substitute for the full run.

## Analysis and Evidence Rules

- Preserve each reviewer comment verbatim in the response registry.
- Establish the corrected 2D baseline before comparing 3D variants: explicitly bidirectional edges, exactly 10 external neighbors by default, one separately added self-loop, and invariance to observation-row permutation.
- In dedicated graph-construction validation artifacts, report external degree, final degree after graph transforms, physical edge-length distributions, cross-section degree, zero-cross-section-neighbor rate, connected components, and cross-section edge fractions. A latent-embedding comparison may instead reference that validation artifact and report only the independently constructed physical evaluation edges it uses.
- Evaluate major conclusions with comparable spot/cell counts wherever feasible. At minimum, plan comparisons for section-wise 2D, pooled-without-cross-section edges, slice-order permutation, measured-z 3D, density/scale alternatives, and graph-propagation removal.
- Separate confirmed evidence, inference, planned experiments, limitations, and unresolved choices. Never report a planned analysis as completed.
- Treat source datasets as read-only. Place new code, tests, logs, tables, and figures under project-controlled paths and attach them to the relevant SQLite task.
- Reuse existing relevant work where verified, including commits `93e47ec`, `311801d`, `00701e6`, `86e392f`, and `6f88462`, but do not assume a commit fully answers a reviewer comment without an artifact-level audit.
- Match comparison outputs by globally unique observation ID, never by row position.
- Read only the embeddings and metadata required for a comparison; do not materialize expression matrices unnecessarily.
- Attribute a difference causally to graph construction only when data, preprocessing, training objectives, annotations, model settings, seeds, and evaluation populations are matched or the remaining differences are explicitly controlled.
- Define every label-neighborhood metric explicitly. Treat training-label separation as training fit, not independent validation; treat post-training labels as internal structure summaries unless an external validation design is present.
- Report latent-space agreement, annotation-neighborhood structure, section mixing, and physical-edge continuity separately. Higher physical-edge cosine may also indicate oversmoothing and is not sufficient evidence of biological accuracy.
- Construct physical evaluation edges independently of the learned embedding. Infer each focal radius from the configured external within-section-neighbor scale, accept cross-section candidates only within that physical radius, report the zero-cross-section-neighbor rate, and never force evaluation edges across gaps.

## Text Rules

- Draft responses point by point and keep claims proportional to direct evidence.
- Use a `.author-response` block with the stable `response-id` when a rebuttal source is created.
- State the exact graph semantics in plain language before equations or implementation details.
- Do not describe the current section-wise graph plus later matching as a unified 3D GNN unless cross-section message-passing edges are actually present.
- Reconcile Methods, configuration, code, and experiment metadata for every stated neighbor count.
- When writing percentage changes, always report the original value, new value, absolute percentage-point change when applicable, and relative change. Do not conflate percent with percentage points.
- Manuscript/SI changes require the Rebuttal Dragon finding, target-map, proposal, explicit approval, apply, and verification sequence.

## Figure Rules

- Use simple geometric diagrams and direct labels. Avoid decorative, exaggerated, or text-heavy illustrations.
- Unless the user or an approved analysis specification explicitly requires sampling, every visualization must include all in-scope observations. Never downsample a plot for speed, memory, file size, visual density, or convenience. If full rendering is computationally difficult, optimize the implementation or request evidence-based resources; do not silently substitute a sample. State any explicitly authorized sampling in the figure caption and audit metadata.
- A graph-construction figure must show 10 external within-section neighbors conceptually, while depicting the self-loop separately.
- Never depict a fixed number of cross-section neighbors unless that number is produced by the stated density-and-distance rule for the illustrated focal spot.
- Encode within-section and cross-section edges with visibly different styles and provide physical distances or scale bars where meaningful.
- Raster figures must be at least 300 DPI at their final physical display size. Export PNGs at 320 DPI to avoid the pixels-per-metre rounding that makes a nominal 300-DPI PNG inspect as `299.9994`; verify the stored metadata and pixel dimensions after generation. Keep PDF outputs vector where possible.
- DPI metadata alone is insufficient. Verify effective DPI after placement using source pixels divided by the final display size, and inspect the actual delivered artifact rather than assuming the source setting survived conversion.
- Plot every UMAP panel through Scanpy's `scanpy.pl.umap` using the stored or precomputed `X_umap` coordinates; do not bypass Scanpy with a raw Matplotlib scatter plot. Record the Scanpy version and the UMAP-coordinate settings in the analysis audit.
- Lark DOCX import can preserve the original image bytes while displaying an inline image in a narrow block. For Lark handoffs, split dense composites into individual panels or attach/link the original PNG and vector PDF when the inline preview is unreadable; do not duplicate the same plot under resolution- or software-based headings.
- Do not overwrite an approved rebuttal figure artifact. Register new or changed rebuttal-figure provenance through the Rebuttal Dragon figure-map workflow. Operational analysis figures that have not been promoted into the rebuttal remain attached to their SQLite workflow task and do not require figure-map registration yet.

## Render and Delivery Rules

Rendering and delivery are intentionally unconfigured at project setup. Before the first response-text, figure, manuscript/SI, or delivery task reaches verification, resolve and record:

- `RENDER_COMMAND`
- `LOCAL_DELIVERABLE`
- optional `SYNC_COMMAND`
- optional `REMOTE_VERIFY_COMMAND`
- `MS_SI_PROJECT_ROOT` when manuscript/SI changes are required

Do not infer these paths or commands from another repository or from the prior private Lark demonstration.

An operational Lark upload of an analysis report does not configure the official render/sync pipeline. Verify Lark documents after import, including titles, image tokens, retained source dimensions, and inline display sizes; successful upload alone is not visual-quality verification.
