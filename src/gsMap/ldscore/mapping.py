
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import pyranges as pr


def create_snp_feature_map(
    bim_df: pd.DataFrame,
    mapping_type: str,
    mapping_data: Union[pd.DataFrame, Dict[str, str]],
    window_size: int = 0,
    strategy: str = "score",
) -> tuple[np.ndarray, int]:

    m_ref = len(bim_df)

    # 1. Define Features
    if mapping_type == 'dict':
        # Direct mapping: RSID -> Feature
        # mapping_data is dict {rsid: feature_name}
        unique_features = sorted(list(set(mapping_data.values())))
        feature_to_idx = {f: i for i, f in enumerate(unique_features)}
        F = len(unique_features)

        mapping_vec = np.full(m_ref, F, dtype=np.int32)

        # Vectorize mapping
        # intersection of bim RSIDs and dict keys
        # This part can be slow, optimize with pandas
        bim_df['feature_idx'] = bim_df['SNP'].map(mapping_data).map(feature_to_idx).fillna(F)
        mapping_vec = bim_df['feature_idx'].values.astype(np.int32)

        return mapping_vec, F

    elif mapping_type == 'bed':
        # Spatial mapping
        # mapping_data is a DataFrame [Feature, Chrom, Start, End, Score, Strand]
        features_df = mapping_data.copy()
        unique_features = features_df['Feature'].unique()
        feature_to_idx = {f: i for i, f in enumerate(unique_features)}
        F = len(unique_features)

        # Convert BIM to PyRanges
        # PyRanges requires Chromosome, Start, End columns
        bim_for_pr = bim_df.rename(columns={"CHR": "Chromosome", "BP": "Start"}).copy()
        bim_for_pr['End'] = bim_for_pr['Start'] + 1  # SNPs are point locations
        bim_pr = pr.PyRanges(bim_for_pr)

        # Prepare Feature PyRanges with Window
        # Ensure proper column names for PyRanges
        if 'Chrom' in features_df.columns:
            features_df = features_df.rename(columns={'Chrom': 'Chromosome'})

        features_df = features_df.copy()  # Avoid modifying original
        features_df['Start'] = features_df['Start'] - window_size
        features_df['End'] = features_df['End'] + window_size
        feat_pr = pr.PyRanges(features_df)

        # Join
        joined = bim_pr.join(feat_pr).df

        # Apply Strategy
        if strategy == 'score' and 'Score' in joined.columns:
            # Keep row with max score for each SNP
            joined = joined.sort_values('Score', ascending=False).drop_duplicates('SNP')
        elif strategy == 'tss' and 'Strand' in joined.columns:
            # Calculate distance to TSS based on strand
            # Logic placeholder...
            pass

        # Map to indices
        snp_to_feat = joined.set_index('SNP')['Feature'].map(feature_to_idx)

        # Fill full vector
        # Ensure alignment with original BIM index
        full_map = pd.Series(np.full(m_ref, F), index=bim_df['SNP'])
        full_map.update(snp_to_feat)

        return full_map.values.astype(np.int32), F

    raise ValueError(f"Unknown mapping type: {mapping_type}")