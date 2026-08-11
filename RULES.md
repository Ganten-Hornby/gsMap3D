# gsMap3D 3D-Graph Revision Rules

## Project Configuration

- `PROJECT_ROOT`: `/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning`
- `WORKFLOW_DB`: `/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/REBUTTAL_WORKFLOW.sqlite`
- `REBUTTAL_SOURCE`: `/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/rebuttal/rebuttal.qmd`
- Source review plan: `/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/comments_response_plan.docx`
- Source review plan SHA-256: `b9660d97ba8fc3777c5cfa2655bd9af5270a5d85c0780c239e94febe3671edf8`
- Working branch: `3d-graph-learning`
- Authoritative revision worktree: `/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning`. Keep all revision code, analyses, logs, and artifacts in this worktree; do not create ad hoc alternative revision worktrees, use opaque scratch aliases such as `meta_lcl`, or place revision edits in the main worktree.
- Primary dataset: `/storage/yangjianLab/lilounan/data/3dst/Fig4to6_3DHeart.h5ad`
- Mouse E9.5 source dataset: `/storage/yangjianLab/lilounan/data/3dst/E9.5/Mouse_E9.5_embryo.h5ad`
- Mouse E9.5 prepared section-aware input: `analysis/revision_graph_3d/inputs/Mouse_E9.5_embryo_sectioned.h5ad`
- Mouse E9.5 physical-metadata audit: `analysis/revision_graph_3d/mouse_e95_physical_metadata_audit.md`
- Manuscript-analysis provenance root: `/storage/yangjianLab/chenwenhao/01_Project/01_Research/202312_gsMap/experiment/20260107_gsmap_3d_report_all_collected`
- Local collected-analysis root used by this revision: `/storage/yangjianLab/lilounan/gsMap3D-scripts/20260107_gsmap_3d_report_all_collected`
- Original Mouse E9.5 section-wise 2D latent outputs: `/storage/yangjianLab/lilounan/gsMap3D-scripts/20260107_gsmap_3d_report_all_collected/mouse_embryo/E9.5/find_latent_representations`
- Section-aware Mouse E9.5 3D latent output: `analysis/revision_graph_3d/runs/mouse_e95_section_aware_3d_k10/find_latent_representations/Mouse_E9.5_embryo_sectioned_add_latent.h5ad`
- Mouse E9.5 comparison artifacts: `analysis/revision_graph_3d/comparisons/mouse_e95_2d_vs_3d`
- Mouse E11.5 heart source dataset: `/storage/yangjianLab/lilounan/data/3dst/Mouse_E11.5_heart.h5ad`
- Matched Mouse E11.5 heart outputs: `analysis/revision_graph_3d/runs/mouse_e115_heart_sectionwise_2d_k10` and `analysis/revision_graph_3d/runs/mouse_e115_heart_section_aware_3d_k10`
- Mouse E11.5 heart comparison artifacts: `analysis/revision_graph_3d/comparisons/mouse_e115_heart_2d_vs_3d`
- Preferred image-containing Lark analysis report: `https://ucn9sfonejxs.feishu.cn/docx/Ria5dWAAQoiEB8xKcAgcFL1ansM#doxcnYWYUf08rSeuTXcCjsh6cUb`. It contains one `Visualizations` section with exactly one metrics image (`L3upb4Bydo6OlKxPHrXccPHSnPh`) and one `heart_region` UMAP image (`X9fibiCuDoK9DjxNI7zcvvU7nPf`), displayed at 1000 pixels wide with neutral reader-facing headings, plus the corresponding vector PDFs as attachments. The 3,717- and 4,835-pixel-wide PNG source bytes and both PDFs were downloaded from Lark and verified byte-identical to the local artifacts after replacement.
- Native-Markdown text copy: `https://ucn9sfonejxs.feishu.cn/file/Avrfb8VUYoo1cpxWXWocm8KMnze` (file token `Avrfb8VUYoo1cpxWXWocm8KMnze`). It is intentionally text-only because a native Drive Markdown file does not bundle local image assets; do not present it as the visual report. Version `7672651113581038822` was verified identical to the local heart-region report (SHA-256 `8de5afd62db58d8df0754fb1a83d9b0a8328d43b41035bf10c0999687aaf78dc`).
- The manuscript/SI project root, render command, local deliverable, and sync target are not configured yet. Do not apply manuscript edits, render, or sync until those values are resolved.

`REBUTTAL_WORKFLOW.sqlite` is the only task and dependency ledger. Do not create or maintain a Markdown task queue.

## Worktree, Environment, and Slurm Rules

- Use the existing `${PROJECT_ROOT}/.venv`; do not create another virtual environment for revision work or for CUDA troubleshooting.
- Keep PyTorch, DGL, and CUDA dependencies compatible with CUDA 12. The environment verified in this session uses PyTorch `2.11.0+cu128` with `torch.version.cuda == 12.8`. Do not introduce a CUDA 13 dependency unless a concrete package requires it, the requirement is verified on a GPU node, and the user explicitly approves the environment change.
- A login node is not evidence for CUDA or DGL GPU compatibility. Run CUDA health checks, GPU imports, and GPU smoke tests inside an allocated GPU node.
- Submit computational workloads through Slurm with callback coverage; do not leave a raw `sbatch` job without a registered callback. Use scheduler/resource-database evidence for CPU and memory requests.
- Omit a Codex-imposed `--time` limit unless the user explicitly asks for one or the scheduler requires it. Cluster or QoS defaults may still apply.
- For repeated OOM failures, continue without asking the user to reconfirm the same recovery policy: raise memory using accounting evidence or implement a memory-bounded code fix. If memory use follows the allocation ceiling or a second enlarged request also OOMs, prefer a code fix over further blind escalation.
- The successful sparse Mouse E9.5 embedding used 42.44 GiB host RAM on one L40S; use 64 GiB as the initial evidence-based memory request for an equivalent full embedding. The full 646,893-spot comparison completed as job `11903866` with 16.83 GiB peak host RAM; use 24 GiB for an equivalent full-celltype comparison. The exact full Mouse E11.5 heart comparison completed as job `11905571` with 2.08 GiB peak host RAM; use 4 GiB for an equivalent 98,966-spot comparison. Recalculate when workload shape changes.

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
- A full exact-shape Mouse E9.5 run is required after a propagation-memory change. Synthetic parity and allocation-regression tests are necessary but do not substitute for the full run.

## Analysis and Evidence Rules

- Preserve each reviewer comment verbatim in the response registry.
- Establish the corrected 2D baseline before comparing 3D variants: explicitly bidirectional edges, exactly 10 external neighbors by default, one separately added self-loop, and invariance to observation-row permutation.
- In dedicated graph-construction validation artifacts, report external degree, final degree after graph transforms, physical edge-length distributions, cross-section degree, zero-cross-section-neighbor rate, connected components, and cross-section edge fractions. A latent-embedding comparison may instead reference that validation artifact and report only the independently constructed physical evaluation edges it uses.
- Evaluate major conclusions with comparable spot/cell counts wherever feasible. At minimum, plan comparisons for section-wise 2D, pooled-without-cross-section edges, slice-order permutation, measured-z 3D, density/scale alternatives, and graph-propagation removal.
- Separate confirmed evidence, inference, planned experiments, limitations, and unresolved choices. Never report a planned analysis as completed.
- Treat source datasets as read-only. Place new code, tests, logs, tables, and figures under project-controlled paths and attach them to the relevant SQLite task.
- Reuse existing relevant work where verified, including commits `93e47ec`, `311801d`, `00701e6`, `86e392f`, and `6f88462`, but do not assume a commit fully answers a reviewer comment without an artifact-level audit.

## Mouse E9.5 Embedding-Comparison Rules

- Match section-wise 2D and section-aware 3D outputs by globally unique observation ID, never by row position. The current comparison verified exact matches for all 646,893 spots, section labels, `heart_region`, `transf_anno_E95`, and 3D coordinates.
- Read only the required H5AD embeddings and metadata for comparisons; do not materialize the expression matrices. The current comparison uses selective `h5py` reads and a stratified 30,000-spot metric sample.
- Treat the uploaded original 2D versus current 3D result as descriptive, not as a graph-only ablation. Besides graph construction, the runs differ in training annotation (`mapped_celltype` versus `transf_anno_E95`), class count (54 versus 32), gene space (13,581 versus 17,634 common/used genes), and two-stage training (`True` versus `False`). A matched-settings 2D rerun is required before assigning differences causally to the 3D graph.
- Define heart-region purity at `k=15` as the mean, across focal observations with `heart_region != "NA"`, of the fraction of the 15 nearest latent neighbors carrying the same non-`NA` `heart_region` label. Restrict both focal observations and the neighbor candidate pool to defined heart regions. It is a neighborhood-structure statistic, not prediction accuracy.
- `heart_region` is defined for 44,644 of 646,893 observations (6.90%): 28,931 LV/atrial and 15,713 right/outflow-tract observations. It is not restricted to cardiomyocytes; only 16,082 defined observations (36.0%) have a `mapped_celltype` containing `cardiomyocytes`.
- When reporting the observed full-data `emb_niche` heart-region-purity change, state both endpoints and both scales: `60.84% → 77.62%`, an absolute increase of `16.77` percentage points and a relative increase of `27.6%`. Do not imply causal improvement because the runs differ in graph construction and other training settings.
- For the matched Mouse E11.5 heart experiment, both arms use all 98,966 observations, `heart_regions` as the explicit two-stage training annotation, and otherwise matched settings. Treat heart-region purity and separation as training-label fit, not independent validation. The current runs share seed, observation order, and model structure, so their train/validation split is expected to match, but split membership was not persisted and must not be described as independently audited.
- Report latent-space agreement (for example linear CKA, Procrustes similarity, and kNN overlap), annotation-neighborhood structure, section mixing, and physical-edge continuity separately. Higher physical-edge cosine may also indicate oversmoothing and is not sufficient evidence of biological accuracy.
- Construct the physical evaluation graph independently of the learned embedding. Infer each focal radius from its 10th external within-section neighbor, accept adjacent-section candidates only within that radius, and report the zero-cross-section-neighbor rate; never force evaluation edges across gaps.

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
- Lark DOCX import can preserve the original image bytes while displaying an inline image in a narrow block. In the current report the 3,717- and 4,835-pixel-wide PNGs are displayed at 1000 pixels wide, and their original bytes plus vector PDFs are retained. For Lark handoffs, split dense composites into individual panels or attach/link the original PNG and vector PDF if a 1000-pixel preview is still unreadable; do not duplicate the same plot under resolution- or software-based headings.
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
