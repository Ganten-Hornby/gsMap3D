import numpy as np
import pandas as pd
import pyranges as pr
from typing import Union, Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_snp_feature_map(
        bim_df: pd.DataFrame,
        mapping_type: str,
        mapping_data: Union[pd.DataFrame, Dict[str, str]],
        window_size: int = 0,
        strategy: str = "score",
) -> Tuple[np.ndarray, List[str]]:
    """
    Create a mapping vector assigning each SNP in the BIM file to a feature index.

    Returns the mapping vector AND the list of feature names corresponding to indices.

    Feature Indexing Scheme
    -----------------------
    - Mapped features get indices: 0, 1, 2, ..., F-1
    - Unmapped SNPs get index: F
    - Total feature indices used: F + 1

    Parameters
    ----------
    bim_df : pd.DataFrame
        PLINK BIM file data (must have 'CHR', 'SNP', 'BP').
    mapping_type : str
        'bed' or 'dict'.
    mapping_data : Union[pd.DataFrame, Dict[str, str]]
        Mapping source data.
    window_size : int
        Window extension (for 'bed').
    strategy : str
        'score' or 'tss'.

    Returns
    -------
    mapping_vec : np.ndarray
        Array of shape (M,) containing the feature index for each SNP.
    feature_names : List[str]
        List of feature names corresponding to indices 0 to F-1.
    """
    m_ref = len(bim_df)
    unique_feature_names = []

    # === STRATEGY A: Dictionary Mapping ===
    if mapping_type == 'dict':
        if not isinstance(mapping_data, dict):
            raise ValueError("mapping_data must be a dictionary when mapping_type='dict'")

        # 1. Identify all unique features
        unique_feature_names = sorted(list(set(mapping_data.values())))
        feature_to_idx = {f: i for i, f in enumerate(unique_feature_names)}
        n_features = len(unique_feature_names)

        # 2. Map Dictionary Values to Indices
        rsid_to_idx = {k: feature_to_idx[v] for k, v in mapping_data.items()}

        # 3. Apply to BIM DataFrame
        mapped_indices = bim_df['SNP'].map(rsid_to_idx).fillna(n_features)

        mapping_vec = mapped_indices.values.astype(np.int32)
        return mapping_vec, unique_feature_names

    # === STRATEGY B: BED/Genomic Interval Mapping ===
    elif mapping_type == 'bed':
        if not isinstance(mapping_data, pd.DataFrame):
            raise ValueError("mapping_data must be a DataFrame when mapping_type='bed'")

        df_features = mapping_data.copy()

        required_cols = ['Feature', 'Chromosome', 'Start', 'End']
        if not all(c in df_features.columns for c in required_cols):
            raise ValueError(f"BED DataFrame missing required columns: {required_cols}")

        # 1. Identify all unique features
        unique_feature_names = sorted(df_features['Feature'].unique())
        feature_to_idx = {f: i for i, f in enumerate(unique_feature_names)}
        n_features = len(unique_feature_names)

        # 2. Prepare BIM for PyRanges
        # IO.py now ensures standard PLINK columns: CHR, BP, SNP
        bim_pr_df = bim_df[['CHR', 'BP', 'SNP']].rename(columns={
            'CHR': 'Chromosome',
            'BP': 'Start'
        })
        bim_pr_df['End'] = bim_pr_df['Start'] + 1

        bim_pr_df['Chromosome'] = bim_pr_df['Chromosome'].astype(str)
        df_features['Chromosome'] = df_features['Chromosome'].astype(str)

        pr_bim = pr.PyRanges(bim_pr_df)

        # 3. Pre-calculate TSS
        if strategy == 'tss':
            if 'Strand' not in df_features.columns:
                raise ValueError("Strategy 'tss' requires 'Strand' column in mapping data.")
            df_features['RefTSS'] = np.where(
                df_features['Strand'] == '+',
                df_features['Start'],
                df_features['End']
            )

        # 4. Apply Window Size
        df_features['Start'] = np.maximum(0, df_features['Start'] - window_size)
        df_features['End'] = df_features['End'] + window_size

        pr_features = pr.PyRanges(df_features)

        # 5. Join
        joined = pr_bim.join(pr_features).df

        if joined.empty:
            logger.warning("No overlaps found between SNPs and Features.")
            return np.full(m_ref, n_features, dtype=np.int32), unique_feature_names

        # 6. Resolve Conflicts
        if strategy == 'score':
            if 'Score' not in joined.columns:
                raise ValueError("Strategy 'score' requires 'Score' column in mapping data.")
            joined = joined.sort_values(by=['SNP', 'Score'], ascending=[True, False])
            joined = joined.drop_duplicates(subset=['SNP'], keep='first')

        elif strategy == 'tss':
            joined['distance_to_tss'] = np.abs(joined['Start'] - joined['RefTSS'])
            joined = joined.sort_values(by=['SNP', 'distance_to_tss'], ascending=[True, True])
            joined = joined.drop_duplicates(subset=['SNP'], keep='first')

        # 7. Finalize Mapping Vector
        joined['feature_idx'] = joined['Feature'].map(feature_to_idx)
        snp_to_idx = joined.set_index('SNP')['feature_idx']

        # Apply to original BIM
        mapping_vec = bim_df['SNP'].map(snp_to_idx).fillna(n_features).values.astype(np.int32)

        return mapping_vec, unique_feature_names

    else:
        raise ValueError(f"Unknown mapping_type: {mapping_type}")