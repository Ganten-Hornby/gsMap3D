# Mouse E11.5 annotation-supervision embedding ablation

All four arms contain the same 98,966 observations. The supervised arms use `heart_regions` with two-stage training; the no-annotation arms use only the single-stage reconstruction objective. Graph, model, seed, feature, and observation settings are otherwise matched within the available run metadata and logs.

The no-annotation `heart_regions` metrics are independent post-training label-structure summaries. The supervised values measure fit to a training target and are not independent validation.

## State metrics

### `emb_cell`

| Metric | 2D supervised | 3D supervised | 2D no annotation | 3D no annotation |
|---|---:|---:|---:|---:|
| Heart-region kNN purity@15 | 0.5245 | 0.4872 | 0.4749 | 0.4721 |
| ctype_user kNN purity@15 | 0.9050 | 0.9017 | 0.9023 | 0.9018 |
| Adjacent-section share among embedding kNN@15 | 0.0781 | 0.0740 | 0.0753 | 0.0756 |
| Mean cosine on cross-section radius edges | 0.4893 | 0.4823 | 0.3896 | 0.3068 |

### `emb_niche`

| Metric | 2D supervised | 3D supervised | 2D no annotation | 3D no annotation |
|---|---:|---:|---:|---:|
| Heart-region kNN purity@15 | 0.9949 | 0.9975 | 0.7628 | 0.7946 |
| ctype_user kNN purity@15 | 0.9154 | 0.9118 | 0.9149 | 0.9145 |
| Adjacent-section share among embedding kNN@15 | 0.1304 | 0.2198 | 0.1064 | 0.1745 |
| Mean cosine on cross-section radius edges | 0.9684 | 0.9819 | 0.5159 | 0.6856 |

## Pairwise representation agreement

| Representation | Comparison | Linear CKA | Procrustes | kNN overlap@15 |
|---|---|---:|---:|---:|
| `emb_cell` | supervised: 2D vs 3D | 0.7006 | 0.7861 | 0.0338 |
| `emb_cell` | no annotation: 2D vs 3D | 0.9414 | 0.9270 | 0.2855 |
| `emb_cell` | 2D: supervised vs no annotation | 0.5630 | 0.7022 | 0.0639 |
| `emb_cell` | 3D: supervised vs no annotation | 0.4780 | 0.6429 | 0.0463 |
| `emb_niche` | supervised: 2D vs 3D | 0.9573 | 0.9770 | 0.0503 |
| `emb_niche` | no annotation: 2D vs 3D | 0.7450 | 0.8252 | 0.0757 |
| `emb_niche` | 2D: supervised vs no annotation | 0.3044 | 0.4745 | 0.0702 |
| `emb_niche` | 3D: supervised vs no annotation | 0.3755 | 0.5053 | 0.0514 |

## Main observations

- Without annotation supervision, `emb_niche` adjacent-section share rises from 10.64% to 17.45% (+6.81 percentage points; +64.0% relative). Its mean cosine on independently defined cross-section radius edges rises from 0.5159 to 0.6856 (+0.1696 cosine units; +32.9% relative). The same directional effect therefore does not require `heart_regions` as a training label.
- Post-training `heart_regions` purity in the no-annotation `emb_niche` rises from 0.7628 to 0.7946 (+3.18 percentage points; +4.2% relative). This is independent of the training objective but remains an internal annotation-structure summary, not external biological validation.
- The effect is specific to the niche representation. In no-annotation `emb_cell`, adjacent-section share changes only from 7.53% to 7.56% (+0.02 percentage points; +0.3% relative), while cross-section physical-edge cosine decreases from 0.3896 to 0.3068 (-0.0828 cosine units; -21.2% relative).
- No-annotation 2D-versus-3D global agreement is higher for `emb_cell` (CKA 0.9414; kNN overlap@15 0.2855) than for `emb_niche` (CKA 0.7450; kNN overlap@15 0.0757), showing that the 3D graph reorganizes local niche neighborhoods more strongly.

## Evaluation notes

- Label-frequency baselines: heart_regions=0.2535; ctype_user=0.8989.
- Physical evaluation graph: 989,660 within-section and 801,494 cross-section directed edges; no fallback edges were forced.
- Zero qualifying cross-section-neighbor fraction: 19.36%.
- All metrics and figures use all 98,966 observations; no downsampling was used.
- UMAP coordinates were computed jointly only for the two no-annotation arms; the existing supervised comparison remains in its prior report.
