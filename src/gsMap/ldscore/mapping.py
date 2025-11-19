import numpy as np
import pandas as pd
import pyranges as pr
from typing import Union, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def create_snp_feature_map(
        bim_df: pd.DataFrame,
        mapping_type: str,
        mapping_data: Union[pd.DataFrame, Dict[str, str]],
        window_size: int = 0,
        strategy: str = "score",
) -> Tuple[np.ndarray, int]:
    """
    Create a mapping vector assigning each SNP in the BIM file to a feature index.

    Parameters
    ----------
    bim_df : pd.DataFrame
        PLINK BIM file data with columns ['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'].
    mapping_type : str
        'bed' for genomic interval mapping or 'dict' for direct RSID mapping.
    mapping_data : Union[pd.DataFrame, Dict[str, str]]
        If 'bed': DataFrame with ['Feature', 'Chromosome', 'Start', 'End', (Score, Strand)].
        If 'dict': Dictionary {rsid: feature_name}.
    window_size : int
        Window extension in base pairs (only for 'bed' type).
    strategy : str
        Conflict resolution strategy for 'bed' type:
        'score' (highest score) or 'tss' (nearest TSS based on strand).

    Returns
    -------
    mapping_vec : np.ndarray
        Array of shape (M,) containing the feature index for each SNP.
        Unmapped SNPs are assigned the index F (where F is the number of features).
    n_features : int
        Total number of unique features (F).
    """
    m_ref = len(bim_df)

    # === STRATEGY A: Dictionary Mapping ===
    if mapping_type == 'dict':
        if not isinstance(mapping_data, dict):
            raise ValueError("mapping_data must be a dictionary when mapping_type='dict'")

        # 1. Identify all unique features
        unique_features = sorted(list(set(mapping_data.values())))
        feature_to_idx = {f: i for i, f in enumerate(unique_features)}
        n_features = len(unique_features)

        # 2. Map Dictionary Values to Indices
        # Create a fast lookup: RSID -> Feature Index
        rsid_to_idx = {k: feature_to_idx[v] for k, v in mapping_data.items()}

        # 3. Apply to BIM DataFrame
        # map() will handle looking up RSIDs. fillna(n_features) assigns the "garbage bin" to unknown SNPs.
        mapped_indices = bim_df['SNP'].map(rsid_to_idx).fillna(n_features)

        mapping_vec = mapped_indices.values.astype(np.int32)
        return mapping_vec, n_features

    # === STRATEGY B: BED/Genomic Interval Mapping ===
    elif mapping_type == 'bed':
        if not isinstance(mapping_data, pd.DataFrame):
            raise ValueError("mapping_data must be a DataFrame when mapping_type='bed'")

        df_features = mapping_data.copy()

        # Validate required columns
        required_cols = ['Feature', 'Chromosome', 'Start', 'End']
        if not all(c in df_features.columns for c in required_cols):
            raise ValueError(f"BED DataFrame missing required columns: {required_cols}")

        # 1. Identify all unique features
        unique_features = sorted(df_features['Feature'].unique())
        feature_to_idx = {f: i for i, f in enumerate(unique_features)}
        n_features = len(unique_features)

        # 2. Prepare BIM for PyRanges
        # Rename to standard PyRanges columns
        bim_pr_df = bim_df[['CHR', 'BP', 'SNP']].rename(columns={
            'CHR': 'Chromosome',
            'BP': 'Start'
        })
        bim_pr_df['End'] = bim_pr_df['Start'] + 1

        # Ensure Chromosomes are strings to prevent type mismatch during join
        bim_pr_df['Chromosome'] = bim_pr_df['Chromosome'].astype(str)
        df_features['Chromosome'] = df_features['Chromosome'].astype(str)

        pr_bim = pr.PyRanges(bim_pr_df)

        # 3. Pre-calculate TSS if needed (before window expansion modifies Start/End)
        if strategy == 'tss':
            if 'Strand' not in df_features.columns:
                raise ValueError("Strategy 'tss' requires 'Strand' column in mapping data.")

            # If '+', TSS is Start. If '-', TSS is End.
            # We store this as a metadata column 'RefTSS'
            df_features['RefTSS'] = np.where(
                df_features['Strand'] == '+',
                df_features['Start'],
                df_features['End']
            )

        # 4. Apply Window Size
        df_features['Start'] = np.maximum(0, df_features['Start'] - window_size)
        df_features['End'] = df_features['End'] + window_size

        pr_features = pr.PyRanges(df_features)

        # 5. Join (Intersect SNPs with Feature Windows)
        # Result contains all columns from both BIM and Feature DF
        # Suffixes are added to Feature columns if names collide (Start_b, End_b, etc.)
        joined = pr_bim.join(pr_features).df

        if joined.empty:
            logger.warning("No overlaps found between SNPs and Features.")
            return np.full(m_ref, n_features, dtype=np.int32), n_features

        # 6. Resolve Conflicts (One SNP mapping to multiple Features)

        if strategy == 'score':
            if 'Score' not in joined.columns:
                raise ValueError("Strategy 'score' requires 'Score' column in mapping data.")

            # Sort by SNP (grouping) and Score (descending)
            joined = joined.sort_values(by=['SNP', 'Score'], ascending=[True, False])

            # Keep only the highest scoring feature for each SNP
            joined = joined.drop_duplicates(subset=['SNP'], keep='first')

        elif strategy == 'tss':
            # Calculate distance from SNP Position (Start) to Reference TSS (RefTSS)
            # Note: In the joined DF, 'Start' comes from the BIM (SNP position).
            joined['distance_to_tss'] = np.abs(joined['Start'] - joined['RefTSS'])

            # Sort by SNP (grouping) and Distance (ascending)
            joined = joined.sort_values(by=['SNP', 'distance_to_tss'], ascending=[True, True])

            # Keep only the nearest feature for each SNP
            joined = joined.drop_duplicates(subset=['SNP'], keep='first')

        # 7. Finalize Mapping Vector
        # Map resulting Feature names to Indices
        joined['feature_idx'] = joined['Feature'].map(feature_to_idx)

        # Create a lookup series: SNP -> Feature Index
        snp_to_idx = joined.set_index('SNP')['feature_idx']

        # Apply to original BIM to ensure alignment and handle unmapped SNPs
        # fillna(n_features) handles SNPs that didn't intersect anything
        mapping_vec = bim_df['SNP'].map(snp_to_idx).fillna(n_features).values.astype(np.int32)

        return mapping_vec, n_features

    else:
        raise ValueError(f"Unknown mapping_type: {mapping_type}")