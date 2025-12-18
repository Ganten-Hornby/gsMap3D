import numpy as np
import pandas as pd
import pyranges as pr
import scipy.sparse
from typing import Union, Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def create_snp_feature_map(
        bim_df: pd.DataFrame,
        mapping_type: str,
        mapping_data: Union[pd.DataFrame, Dict[str, str]],
        feature_window_size: int = 0,
        strategy: str = "score",
) -> Tuple[scipy.sparse.csr_matrix, List[str], Optional[pd.DataFrame]]:
    """
    Create a sparse mapping matrix assigning each SNP in the BIM file to feature indices.

    Returns:
    1. Sparse matrix where rows correspond to SNPs (in bim_df order) and columns correspond to features.
    2. List of feature names corresponding to indices 0 to F-1.
    3. (Optional) DataFrame containing the curated SNP-feature mappings (for BED type).

    Feature Indexing Scheme
    -----------------------
    - Mapped features get indices: 0, 1, 2, ..., F-1
    - Unmapped bin gets index: F
    - Total columns: F + 1
    - Values: Score from BED file (if available, else 1.0)

    Parameters
    ----------
    bim_df : pd.DataFrame
        PLINK BIM file data (must have 'CHR', 'SNP', 'BP').
    mapping_type : str
        'bed' or 'dict'.
    mapping_data : Union[pd.DataFrame, Dict[str, str]]
        Mapping source data. For 'bed' type, this should be a DataFrame read from
        a standard BED file using pr.read_bed() with columns:
        - Chromosome (chr)
        - Start (0-based start position)
        - End (1-based end position)
        - Feature (name/identifier from 4th column)
        - Score (optional, from 5th column)
        - Strand (optional, from 6th column)

        Note: BED files should be in standard BED6 format WITHOUT a header line.
    feature_window_size : int
        Window extension (for 'bed').
    strategy : str
        'score', 'tss', 'center', or 'allow_repeat'.
        - 'allow_repeat': A SNP can map to multiple features (values are summed or kept).
        - 'score': Keep mapping with highest score per SNP.
        - 'tss': Keep mapping closest to TSS per SNP.
        - 'center': Keep mapping closest to the center of the feature interval per SNP.

    Returns
    -------
    mapping_matrix : scipy.sparse.csr_matrix
        Sparse matrix of shape (M, F+1).
    feature_names : List[str]
        List of feature names corresponding to indices 0 to F-1.
    mapping_df : Optional[pd.DataFrame]
        DataFrame with columns [SNP, Feature, ...] showing final mappings. None if using 'dict'.
    """
    m_ref = len(bim_df)
    unique_feature_names = []
    curated_mapping_df = None

    # Prepare basic SNP info for joining
    # We assign an integer index to every SNP in BIM to construct the sparse matrix later
    bim_df = bim_df.copy()
    bim_df['snp_row_idx'] = np.arange(m_ref)

    # Intermediate storage for sparse construction
    row_indices = []
    col_indices = []
    data_values = []

    # === STRATEGY A: Dictionary Mapping ===
    if mapping_type == 'dict':
        if not isinstance(mapping_data, dict):
            raise ValueError("mapping_data must be a dictionary when mapping_type='dict'")

        # 1. Identify all unique features
        unique_feature_names = sorted(list(set(mapping_data.values())))
        feature_to_idx = {f: i for i, f in enumerate(unique_feature_names)}
        n_features = len(unique_feature_names)

        # 2. Map Dictionary Values to Indices
        # Filter mapping data to only include SNPs present in BIM to save memory/time
        bim_snps = set(bim_df['SNP'])
        valid_mapping = {k: v for k, v in mapping_data.items() if k in bim_snps}

        # 3. Create Sparse Entries
        # Create a Series for mapping: SNP -> Feature Index
        snp_to_feat_idx = pd.Series(valid_mapping).map(feature_to_idx)

        # Map BIM SNPs to feature indices
        mapped_feat_indices = bim_df['SNP'].map(snp_to_feat_idx)

        # Drop NaNs (Unmapped)
        valid_mask = mapped_feat_indices.notna()

        if valid_mask.any():
            row_indices = bim_df.loc[valid_mask, 'snp_row_idx'].values
            col_indices = mapped_feat_indices[valid_mask].values.astype(int)
            data_values = np.ones(len(row_indices), dtype=np.float32)

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
        bim_pr_df = bim_df[['CHR', 'BP', 'SNP', 'snp_row_idx']].rename(columns={
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
        df_features['Start'] = np.maximum(0, df_features['Start'] - feature_window_size)
        df_features['End'] = df_features['End'] + feature_window_size

        pr_features = pr.PyRanges(df_features)

        # 5. Join
        # Columns from pr_features (Right) that overlap with pr_bim (Left) get suffix '_b'
        # Overlapping cols usually: Start, End.
        # pr_bim (SNP) Start is 'Start'. pr_features (Window) Start is 'Start_b'.
        joined = pr_bim.join(pr_features, apply_strand_suffix=False).df

        if not joined.empty:
            # 6. Resolve Conflicts / Filter
            if strategy == 'score':
                if 'Score' not in joined.columns:
                    raise ValueError("Strategy 'score' requires 'Score' column in mapping data.")
                joined = joined.sort_values(by=['SNP', 'Score'], ascending=[True, False])
                joined = joined.drop_duplicates(subset=['SNP'], keep='first')

            elif strategy == 'tss':
                joined['distance_to_tss'] = np.abs(joined['Start'] - joined['RefTSS'])
                joined = joined.sort_values(by=['SNP', 'distance_to_tss'], ascending=[True, True])
                joined = joined.drop_duplicates(subset=['SNP'], keep='first')

            elif strategy == 'center':
                # Calculate center of the feature interval (the window around feature)
                # 'Start_b' and 'End_b' are the feature window coordinates from PyRanges join
                joined['feature_center'] = (joined['Start_b'] + joined['End_b']) / 2.0
                # Calculate distance from SNP position ('Start') to center
                joined['distance_to_center'] = np.abs(joined['Start'] - joined['feature_center'])
                # Sort and pick closest
                joined = joined.sort_values(by=['SNP', 'distance_to_center'], ascending=[True, True])
                joined = joined.drop_duplicates(subset=['SNP'], keep='first')

            elif strategy == 'allow_repeat':
                # No de-duplication. One SNP can map to multiple features.
                pass

            # 7. Prepare Sparse Data
            # Row indices come from BIM 'snp_row_idx' which is preserved in join
            row_indices = joined['snp_row_idx'].values
            col_indices = joined['Feature'].map(feature_to_idx).values

            # Data values: Use Score if available and requested, otherwise 1.0
            if 'Score' in joined.columns and strategy in ['score', 'allow_repeat']:
                # Use provided score
                data_values = joined['Score'].values.astype(np.float32)
            else:
                # Default to 1.0 for geometric strategies (tss, center) or missing score
                data_values = np.ones(len(row_indices), dtype=np.float32)

            # Save the result for output
            # We filter columns to make it cleaner
            output_cols = ['SNP', 'Chromosome', 'Start', 'Feature']
            if 'Score' in joined.columns: output_cols.append('Score')
            if 'distance_to_tss' in joined.columns: output_cols.append('distance_to_tss')
            if 'distance_to_center' in joined.columns: output_cols.append('distance_to_center')

            # Ensure columns exist before selecting
            output_cols = [c for c in output_cols if c in joined.columns]
            curated_mapping_df = joined[output_cols].copy()
            curated_mapping_df.rename(columns={'Start': 'SNP_BP'}, inplace=True)

    else:
        raise ValueError(f"Unknown mapping_type: {mapping_type}")

    # === Final Matrix Construction ===

    # Determine Unmapped SNPs
    # We create a matrix of shape (M, F+1). The last column (index F) is for unmapped.
    # Any row_idx NOT present in row_indices gets a 1.0 in the last column.

    mapped_rows_set = set(row_indices)
    all_rows = set(range(m_ref))
    unmapped_rows = list(all_rows - mapped_rows_set)

    if unmapped_rows:
        unmapped_rows = np.array(unmapped_rows, dtype=int)
        unmapped_cols = np.full(len(unmapped_rows), n_features, dtype=int)  # Last column index
        unmapped_data = np.ones(len(unmapped_rows), dtype=np.float32)

        # Append unmapped data
        if len(row_indices) > 0:
            row_indices = np.concatenate([row_indices, unmapped_rows])
            col_indices = np.concatenate([col_indices, unmapped_cols])
            data_values = np.concatenate([data_values, unmapped_data])
        else:
            row_indices = unmapped_rows
            col_indices = unmapped_cols
            data_values = unmapped_data

    else:
        # If all SNPs mapped, we technically don't need to append anything,
        # but we still need the matrix shape to include the extra column.
        pass

    # Construct CSR Matrix
    # Shape: (m_ref, n_features + 1)
    mapping_matrix = scipy.sparse.csr_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(m_ref, n_features + 1),
        dtype=np.float32
    )

    return mapping_matrix, unique_feature_names, curated_mapping_df