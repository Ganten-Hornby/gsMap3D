import torch
import scanpy as sc
import sys
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import re 
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeRemainingColumn
from scipy.special import softmax
from .gnn.gcn import GCN, build_spatial_graph
from gsMap.config import FindLatentRepresentationsConfig
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def convert_to_human_genes(adata, gene_homolog_dict: dict, species: str = None):
    """
    Convert gene names in adata to human gene symbols using a pre-computed mapping dictionary.
    
    Args:
        adata: AnnData object to convert
        gene_homolog_dict: Dictionary mapping current gene names to human gene symbols
        species: Optional species name to use for the original gene column in var
        
    Returns:
        adata: Processed AnnData object with human symbols as var_names
    """
    # Identify common genes present in mapping
    common_genes = [g for g in gene_homolog_dict.keys() if g in adata.var_names]
    
    if len(common_genes) < 300:
        logger.warning(f"Only {len(common_genes)} genes found in mapping dictionary.")
    
    # Filter to these genes
    adata = adata[:, common_genes].copy()
    
    human_symbols = [gene_homolog_dict[g] for g in adata.var_names]
    
    # Store original gene names if species is provided
    if species:
        species_col_name = f"{species}_GENE_SYM"
        adata.var[species_col_name] = adata.var_names.values
    
    # Update var_names to human symbols
    adata.var_names = human_symbols
    adata.var.index.name = "HUMAN_GENE_SYM"
    
    # Remove duplicated genes (keep first occurrence)
    if adata.var_names.duplicated().any():
        n_before = adata.n_vars
        adata = adata[:, ~adata.var_names.duplicated()].copy()
        logger.info(f"Removed {n_before - adata.n_vars} duplicate human gene symbols.")
        
    return adata

def find_common_hvg(sample_h5ad_dict, params: FindLatentRepresentationsConfig):
    """
    Identifies common highly variable genes (HVGs) across multiple ST datasets and calculates
    the number of cells to sample from each dataset.

    Args:
        sample_h5ad_dict (dict): Dictionary mapping sample names to file paths of ST datasets.
        params (object): Parameter object containing attributes.
    """

    variances_list = []
    cell_number = []

    logger.info("Finding highly variable genes (HVGs)...")

    with Progress(
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Finding HVGs", total=len(sample_h5ad_dict))

        for sample_name, st_file in sample_h5ad_dict.items():
            adata_temp = sc.read_h5ad(st_file)
            # sc.pp.filter_genes(adata_temp, min_counts=1)

            # Filter out mitochondrial and hemoglobin genes
            gene_keep = ~adata_temp.var_names.str.match(re.compile(r'^(HB.-|MT-)', re.IGNORECASE))
            if removed_genes := adata_temp.n_vars - gene_keep.sum():
                logger.info(f"Removed {removed_genes} mitochondrial and hemoglobin genes in {sample_name}.")

            adata_temp = adata_temp[:,gene_keep].copy()

            # Make gene names unique to avoid issues with HVG calculation
            adata_temp.var_names_make_unique()

            is_count_data, actual_data_layer = setup_data_layer(adata_temp, params.data_layer, verbose=False)
            cell_number.append(adata_temp.n_obs)

            try:
                # Identify highly variable genes
                if is_count_data:
                    sc.experimental.pp.highly_variable_genes(
                        adata_temp, n_top_genes=params.feat_cell, subset=False
                    )
                else:
                    sc.pp.highly_variable_genes(
                        adata_temp, n_top_genes=params.feat_cell, subset=False, flavor='seurat'
                    )

                var_df = adata_temp.var.copy()
                var_df["gene"] = var_df.index.tolist()
                variances_list.append(var_df)
            except Exception as e:
                logger.warning(f"[HVG skipped] {e}")
                continue
            

            progress.update(task, advance=1)

    # Check if we have any valid variance results
    if len(variances_list) == 0:
        raise ValueError("No valid HVG results obtained from any sample. Please check your data and parameters.")

    # Find the common genes across all samples
    common_genes = np.array(
        list(set.intersection(
            *map(set, [st.index.to_list() for st in variances_list])))
    )

    # Error when gene number is too small
    if len(common_genes) < 300:
        raise ValueError(f"Only {len(common_genes)} common genes between all samples.")

    # Aggregate variances and identify HVGs
    df = pd.concat(variances_list, axis=0)
    df["highly_variable"] = df["highly_variable"].astype(int)
    
    if is_count_data:
        df = df.groupby("gene", observed=True).agg(
            dict(
                residual_variances="median",
                highly_variable="sum",
            )
        )
        df = df.loc[common_genes]
        df["highly_variable_nbatches"] = df["highly_variable"]
        df.sort_values(
            ["highly_variable_nbatches", "residual_variances"],
            ascending=False,
            na_position="last",
            inplace=True,
        )
    else:
        df = df.groupby("gene", observed=True).agg(
            dict(
                dispersions_norm="median",
                highly_variable="sum",
            )
        )
        df = df.loc[common_genes]
        df["highly_variable_nbatches"] = df["highly_variable"]
        df.sort_values(
            ["highly_variable_nbatches", "dispersions_norm"],
            ascending=False,
            na_position="last",
            inplace=True,
        )

    hvg = df.iloc[: params.feat_cell,].index.tolist()

    # Find the number of sampling cells for each batch
    total_cell = np.sum(cell_number)
    total_cell_training = np.minimum(total_cell, params.n_cell_training)
    cell_proportion = cell_number / total_cell
    n_cell_used = [
        int(cell) for cell in (total_cell_training * cell_proportion).tolist()
    ]

    # Only use the common genes that can be transformed to human genes
    if params.species is not None:
        homologs = pd.read_csv(params.homolog_file, sep='\t')
        if homologs.shape[1] < 2:
            raise ValueError("Homologs file must have at least two columns: one for the species and one for the human gene symbol.")
        homologs.columns = [params.species, 'HUMAN_GENE_SYM']
        homologs.set_index(params.species, inplace=True)
        common_genes = np.intersect1d(common_genes, homologs.index)
        gene_homolog_dict = dict(zip(common_genes,homologs.loc[common_genes].HUMAN_GENE_SYM.values))
    else:
        gene_homolog_dict = dict(zip(common_genes,common_genes))

    if len(gene_homolog_dict) < 300:
        raise ValueError(f"Only {len(gene_homolog_dict)} genes could be mapped to human symbols. Please check the homolog file.")

    return hvg, n_cell_used, gene_homolog_dict


def create_subsampled_adata(sample_h5ad_dict, n_cell_used, params: FindLatentRepresentationsConfig):
    """
    Create subsampled adata for each sample with sample-specific stratified sampling,
    add batch and label information, and return concatenated adata.

    Args:
        sample_h5ad_dict (dict): Dictionary mapping sample names to file paths of ST datasets.
        n_cell_used (list): Number of cells to sample from each dataset.
        params (object): Parameter object containing attributes.

    Returns:
        adata: Concatenated adata object with batch and label information in obs.
    """
    subsampled_adatas = []

    logger.info("Creating subsampled adata with stratified sampling...")

    for st_id, (sample_name, st_file) in enumerate(sample_h5ad_dict.items()):
        logger.info(f"Processing {sample_name}...")

        # Load the data
        adata = sc.read_h5ad(st_file)

        # Filter out mitochondrial and hemoglobin genes
        gene_keep = ~adata.var_names.str.match(re.compile(r'^(HB.-|MT-)', re.IGNORECASE))
        adata = adata[:,gene_keep].copy()

        # Make gene names unique to avoid issues with concatenation
        adata.var_names_make_unique()

        # Setup data layer consistently
        is_count_data, actual_data_layer = setup_data_layer(adata, params.data_layer)

        # Filter cells based on annotation if provided
        if params.annotation is not None:
            adata = adata[~adata.obs[params.annotation].isnull()]

        # Perform stratified sampling within this sample
        if params.do_sampling and adata.n_obs > n_cell_used[st_id]:
            if params.annotation is None:
                # Simple random sampling
                num_cell = n_cell_used[st_id]
                logger.info(f"Downsampling {sample_name} to {num_cell} cells...")
                random_indices = np.random.choice(adata.n_obs, num_cell, replace=False)
                adata = adata[random_indices].copy()
            else:
                # Stratified sampling based on sample-specific annotation distribution
                sample_annotation_counts = adata.obs[params.annotation].value_counts()
                sample_total_cells = len(adata)
                target_total_cells = n_cell_used[st_id]

                # Calculate sample-specific annotation proportions
                sample_annotation_proportions = sample_annotation_counts / sample_total_cells

                # Calculate target cells for each annotation in this sample
                target_cells_per_annotation = (sample_annotation_proportions * target_total_cells).astype(int)

                logger.info(f"Downsampling {sample_name} to {target_total_cells} cells...")
                logger.debug("---Sample-specific annotation distribution-----")
                for ann, count in target_cells_per_annotation.items():
                    logger.debug(f"{ann}: {count} cells")

                # Perform stratified sampling
                sampled_cells = (
                    adata.obs.groupby(params.annotation, group_keys=False, observed=True)
                    .apply(
                        lambda x: x.sample(
                            max(min(target_cells_per_annotation.get(x.name, 0), len(x)), 1),
                            replace=False,
                        ),
                        include_groups=False
                    )
                    .index
                )

                # Filter adata to sampled cells
                adata = adata[sampled_cells].copy()

        # Add batch information to obs
        adata.obs['batch_id'] = f"S{st_id}"
        adata.obs['sample_name'] = sample_name

        # Add label information to obs (ensure it's properly set)
        if params.annotation is not None:
            # The annotation column already exists, just ensure it's called 'label'
            if 'label' not in adata.obs.columns:
                adata.obs['label'] = adata.obs[params.annotation]
        else:
            # Create dummy labels
            adata.obs['label'] = 'unknown'

        subsampled_adatas.append(adata)
        logger.info(f"Processed {sample_name}: {adata.n_obs} cells, {adata.n_vars} genes")

    # Concatenate all samples
    logger.info("Concatenating all processed data...")
    concatenated_adata = sc.concat(subsampled_adatas, axis=0, join='inner',
                                   index_unique='_', fill_value=0)

    logger.info(f"Final concatenated adata: {concatenated_adata.n_obs} cells, {concatenated_adata.n_vars} genes")

    return concatenated_adata

def _looks_like_count_matrix(X, max_check=100, tol=1e-8):
    """
    Heuristically check whether X contains integer values.
    """
    if X is None:
        return False

    if sp.issparse(X):
        data = X.data
        if data.size == 0:
            return False
        if data.size > max_check:
            data = data[:max_check]
    else:
        flat = np.asarray(X).ravel()
        if flat.size == 0:
            return False
        if flat.size > max_check:
            flat = flat[:max_check]
        data = flat

    nz = data[data != 0]
    if nz.size == 0:
        return True  # all zero, good enough
    return np.allclose(nz, np.round(nz), atol=tol)


def setup_data_layer(adata, data_layer, verbose=True):
    """
    Setup and validate data layer for adata object.

    Args:
        adata: AnnData object
        data_layer: Requested data layer name
        verbose: Whether to log information messages (default: True)

    Returns:
        tuple: (is_count_data, actual_data_layer)
            - is_count_data: Boolean indicating if data is count data
            - actual_data_layer: The actual layer name to use ("X" if using adata.X)
    """
    # Check if the requested data layer exists in layers
    if  data_layer in adata.layers:
        # Use the specified layer
        adata.X = adata.layers[data_layer]
        actual_data_layer = data_layer
        if verbose:
            logger.info(f"Using data layer: {data_layer}")
    elif data_layer == "X":
        actual_data_layer = "X"
    else:
        raise ValueError(f"Data layer '{data_layer}' not found in adata.layers.")

    # Determine if this is count data
    is_count_data = actual_data_layer in ["count", "counts", "raw_counts","impute_count"] or \
                   (adata.X is not None and np.issubdtype(adata.X.dtype, np.integer)) or \
                     _looks_like_count_matrix(adata.X)

    if verbose:
        if is_count_data:
            logger.info("Detected count data")
        else:
            logger.info("Data appears to be normalized/log-transformed")

    return is_count_data, actual_data_layer


def normalize_for_analysis(adata, is_count_data, preserve_raw=True):
    """
    Normalize data for DEG analysis or module scoring if needed.

    Args:
        adata: AnnData object
        is_count_data: Boolean indicating if data is count data
        preserve_raw: Whether to preserve raw data in adata.raw

    Returns:
        adata: AnnData object with normalized data
    """
    if is_count_data:
        logger.info("Normalizing and log-transforming count data...")

        # Store raw counts if requested and not already stored
        if preserve_raw and adata.raw is None:
            adata.raw = adata

        # Normalize to 10,000 reads per cell
        sc.pp.normalize_total(adata, target_sum=1e4)

        # Log-transform (log1p)
        sc.pp.log1p(adata)

        logger.info("Data normalized and log-transformed")
    else:
        logger.info("Using data as-is (already normalized/log-transformed)")

    return adata


def filter_significant_degs(deg_results, annotation, adata=None, pval_threshold=0.05, lfc_threshold=0.25, max_genes=100):
    """
    Filter DEGs based on statistical significance and fold change criteria.

    Args:
        deg_results: DEG results from scanpy rank_genes_groups
        annotation: Annotation label to get DEGs for
        adata: Optional adata to check gene existence
        pval_threshold: P-value threshold (default: 0.05)
        lfc_threshold: Log fold change threshold (default: 0.25)
        max_genes: Maximum number of genes to return (default: 100)

    Returns:
        list: Filtered list of significant DEG gene names
    """
    gene_names = deg_results['names'][annotation]
    pvals_adj = deg_results['pvals_adj'][annotation]
    logfoldchanges = deg_results['logfoldchanges'][annotation]

    # Filter genes based on significance and fold change
    annotation_genes = []
    for gene, pval, lfc in zip(gene_names, pvals_adj, logfoldchanges):
        if gene is not None and pval < pval_threshold and abs(lfc) > lfc_threshold:
            annotation_genes.append(gene)

    # Limit to max genes if we have more
    if len(annotation_genes) > max_genes:
        annotation_genes = annotation_genes[:max_genes]

    # Ensure genes exist in adata if provided
    if adata is not None:
        annotation_genes = [gene for gene in annotation_genes if gene in adata.var_names]

    return annotation_genes


def calculate_module_scores_from_degs(adata, deg_results, annotation_key):
    """
    Calculate module scores using existing DEG results.

    Args:
        adata: AnnData object to calculate scores for
        deg_results: DEG results from scanpy rank_genes_groups
        annotation_key: Column name in obs containing annotation labels

    Returns:
        adata: Updated adata with module score columns
    """

    logger.info("Calculating module scores using existing DEG results...")

    # Detect count data and normalize if needed
    is_count_data = _looks_like_count_matrix(adata.X)
    adata = normalize_for_analysis(adata, is_count_data, preserve_raw=False)

    # Ensure annotation is categorical if it exists
    if annotation_key in adata.obs.columns:
        adata.obs[annotation_key] = adata.obs[annotation_key].astype('category')
        available_annotations = adata.obs[annotation_key].cat.categories
    else:
        # If annotation doesn't exist, use all annotations from DEG results
        available_annotations = list(deg_results['names'].dtype.names)

    # Calculate module score for each annotation
    for annotation in available_annotations:
        if annotation in deg_results['names'].dtype.names:
            # Get significant DEGs for this annotation
            annotation_genes = filter_significant_degs(deg_results, annotation, adata)

            if len(annotation_genes) > 0:
                logger.info(f"Calculating module score for {annotation} using {len(annotation_genes)} genes")

                # Calculate module score
                sc.tl.score_genes(
                    adata,
                    gene_list=annotation_genes,
                    score_name=f"{annotation}_module_score",
                    use_raw=False
                )
            else:
                logger.warning(f"No valid DEGs found for {annotation}")
                adata.obs[f"{annotation}_module_score"] = 0.0
        else:
            logger.warning(f"Annotation {annotation} not found in DEG results")
            adata.obs[f"{annotation}_module_score"] = 0.0

    return adata


def calculate_module_score(training_adata, annotation_key):
    """
    Perform DEG analysis for each annotation and calculate module scores.

    Args:
        training_adata: Concatenated training adata with annotation information
        annotation_key: Column name in obs containing annotation labels

    Returns:
        training_adata: Updated adata with module scores for each annotation
    """

    logger.info("Performing DEG analysis for each annotation...")

    # Make a copy to avoid modifying the original
    adata = training_adata.copy()

    # Ensure annotation is categorical
    adata.obs[annotation_key] = adata.obs[annotation_key].astype('category')

    # Detect count data and normalize if needed
    is_count_data = _looks_like_count_matrix(adata.X)
    adata = normalize_for_analysis(adata, is_count_data, preserve_raw=True)

    # Perform DEG analysis with DataFrame fragmentation warnings suppressed
    # These warnings are harmless and come from Scanpy's internal implementation
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning,
                              message='DataFrame is highly fragmented')
        sc.tl.rank_genes_groups(
            adata,
            groupby=annotation_key,
            method='wilcoxon',
            use_raw=False,
            n_genes=100
        )

    logger.info("Calculating module scores for each annotation...")

    # Get DEG results
    deg_results = adata.uns['rank_genes_groups']

    # Calculate module score for each annotation
    for annotation in adata.obs[annotation_key].cat.categories:

        # Get significant DEGs for this annotation
        annotation_genes = filter_significant_degs(deg_results, annotation, adata)

        if len(annotation_genes) > 0:
            logger.info(f"Calculating module score for {annotation} using {len(annotation_genes)} genes")

            # Calculate module score
            sc.tl.score_genes(
                adata,
                gene_list=annotation_genes,
                score_name=f"{annotation}_module_score",
                use_raw=False
            )
        else:
            logger.warning(f"No valid DEGs found for {annotation}")
            adata.obs[f"{annotation}_module_score"] = 0.0

    logger.info("Module score calculation completed")
    return adata


def apply_module_score_qc(adata, annotation_key, module_score_threshold_dict):
    """
    Apply quality control based on module scores.

    Args:
        adata: AnnData object to apply QC to
        annotation_key: Column name in obs containing annotation labels
        module_score_threshold_dict: Dictionary mapping annotation to threshold values

    Returns:
        adata: Updated adata with QC information
    """
    logger.info("Applying module score-based quality control...")

    # Initialize High_quality column as boolean (True = high quality)
    adata.obs['High_quality'] = True

    # Check if we have the annotation key
    if annotation_key not in adata.obs.columns:
        logger.warning(f"Annotation key '{annotation_key}' not found in adata.obs. Skipping module score QC.")
        return adata

    # Apply QC for each annotation
    for annotation, threshold in module_score_threshold_dict.items():
        module_score_col = f"{annotation}_module_score"

        if module_score_col in adata.obs.columns:
            # Find cells of this annotation with low module scores
            annotation_mask = adata.obs[annotation_key] == annotation
            low_score_mask = adata.obs[module_score_col] < threshold

            # Set High_quality to False for cells that match both conditions (low quality)
            low_quality_mask = annotation_mask & low_score_mask
            adata.obs.loc[low_quality_mask, 'High_quality'] = False

            n_low_quality = low_quality_mask.sum()
            n_annotation_cells = annotation_mask.sum()

            logger.info(f"{annotation}: {n_low_quality}/{n_annotation_cells} cells marked as low quality "
                       f"(threshold: {threshold:.3f})")
        else:
            logger.warning(f"Module score column '{module_score_col}' not found in adata.obs")

    total_low_quality = (~adata.obs['High_quality']).sum()
    logger.info(f"Total low quality cells: {total_low_quality}/{adata.n_obs}")

    return adata


# prepare the trainning data
class TrainingData(object):
    """
    Managing and processing training data for graph-based models.

    Attributes:
        params (dict): A dictionary of parameters used for data processing and training.
    """

    def __init__(self, params):
        self.params = params
        self.gcov = GCN(self.params.K)
        self.expression_merge = None
        self.expression_gcn_merge = None
        self.label_merge = None
        self.batch_merge = None
        self.batch_size = None
        self.label_name = None
             
 
    def prepare(self, concatenated_adata, hvg):
        logger.info("Processing concatenated subsampled data...")

        # Get labels from obs
        if self.params.annotation is not None:
            label = concatenated_adata.obs['label'].values
        else:
            label = np.zeros(concatenated_adata.n_obs)

        # Get batch information from obs
        batch_labels = concatenated_adata.obs['batch_id'].values

        # Get expression array for HVG genes
        expression_array = torch.Tensor(concatenated_adata[:, hvg].X.toarray())
        logger.info(f"Expression array shape: {expression_array.shape}")

        # Process each batch separately for GCN (since spatial graphs are sample-specific)
        expression_array_gcn_list = []

        for batch_id in concatenated_adata.obs['batch_id'].unique():
            batch_mask = concatenated_adata.obs['batch_id'] == batch_id
            batch_adata = concatenated_adata[batch_mask]
            batch_expression = expression_array[batch_mask.values]

            logger.info(f"Processing batch {batch_id} with {batch_adata.n_obs} cells...")

            # Build spatial graph for this batch
            edge = build_spatial_graph(
                coords=np.array(batch_adata.obsm[self.params.spatial_key]),
                n_neighbors=self.params.n_neighbors,
            )
            edge = torch.from_numpy(edge.T).long()

            # Apply GCN to this batch
            batch_expression_gcn = self.gcov(batch_expression, edge)
            expression_array_gcn_list.append(batch_expression_gcn)

            logger.info(f"Graph for {batch_id} has {edge.size(1)} edges, {batch_adata.n_obs} cells.")

        # Concatenate GCN results in the same order as the original data
        expression_array_gcn = torch.cat(expression_array_gcn_list, dim=0)

        # Convert batch labels to numeric codes
        batch_codes = pd.Categorical(batch_labels).codes

        # Convert labels to categorical codes
        cat_labels = pd.Categorical(label)
        label_codes = cat_labels.codes

        # Store results
        self.expression_merge = expression_array
        self.expression_gcn_merge = expression_array_gcn
        self.batch_merge = torch.Tensor(batch_codes)
        self.label_merge = torch.Tensor(label_codes).long()

        if self.params.annotation is not None:
            self.label_name = cat_labels.categories.take(np.unique(cat_labels.codes)).to_list()
        else:
            self.label_name = None

        # Set batch size
        self.batch_size = len(torch.unique(self.batch_merge))


# Inference for each ST dataset
class InferenceData(object):
    """
    Infer cell embeddings for each spatial transcriptomics (ST) dataset.
    Attributes:
        hvg: List of highly variable genes.
        batch_size: Integer defining the batch size for inference.
        model: Model to be used for inference.
        params: Dictionary containing additional parameters for inference.
    """

    def __init__(self, hvg, batch_size, model, label_name, params):
        self.params = params
        self.gcov = GCN(self.params.K)
        self.hvg = hvg
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.label_name = label_name
        self.processed_list_path = self.params.latent_dir / 'processed.list'
    

    def infer_embedding_single(self, st_id, st_file) -> Path:
        st_name = (Path(st_file).name).split(".h5ad")[0]
        logger.info(f"Infering cell embeddings for {st_name}...")

        # Load the ST data
        adata = sc.read_h5ad(st_file)
        # sc.pp.filter_genes(adata, min_counts=1)

        # Make gene names unique to avoid reindexing issues
        adata.var_names_make_unique()

        # Setup data layer consistently
        is_count_data, actual_data_layer = setup_data_layer(adata, self.params.data_layer)

        # print(adata.shape)
        # Convert expression data to torch.Tensor
        expression_array = torch.Tensor(adata[:, self.hvg].X.toarray())

        # Graph convolution of expression array
        edge = build_spatial_graph(
            coords=np.array(adata.obsm[self.params.spatial_key]),
            n_neighbors=self.params.n_neighbors,
        )
        edge = torch.from_numpy(edge.T).long()
        expression_array_gcn = self.gcov(expression_array, edge)

        # Build batch vector as one-hot encoding
        n_cell = adata.n_obs
        batch_indices = torch.full((n_cell,), st_id, dtype=torch.long)

        # Prepare the evaluation DataLoader
        dataset = TensorDataset(expression_array_gcn,expression_array, batch_indices)
        Inference_loader = DataLoader(dataset=dataset, batch_size=512, shuffle=False)

        # Inference process
        emb_cell, emb_niche, class_prob = [], [], []

        for (
            expression_gcn_focal,
            expression_focal,
            batch_indices_fcocal,
        ) in Inference_loader:
            expression_gcn_focal = expression_gcn_focal.to(self.device)
            expression_focal = expression_focal.to(self.device)
            batch_indices_fcocal = batch_indices_fcocal.to(self.device)

            self.model.eval()
            with torch.no_grad():
                mu_focal = self.model.encode(
                    [expression_focal, expression_gcn_focal], batch_indices_fcocal
                )
                _,x_class, _, _ = self.model(
                    [expression_focal, expression_gcn_focal], batch_indices_fcocal
                )
                
                class_prob.append(x_class.cpu().numpy())
                emb_cell.append(mu_focal[0].cpu().numpy())
                emb_niche.append(mu_focal[1].cpu().numpy())

        # Concatenate results and store embeddings in adata
        emb_cell = np.concatenate(emb_cell, axis=0)
        emb_niche = np.concatenate(emb_niche, axis=0)
        class_prob = np.concatenate(class_prob, axis=0)
        
        # if self.label_name is not None:
        #     class_prob = pd.DataFrame(softmax(class_prob,axis=1), columns=self.label_name,index=adata.obs_names)
        #     adata.obsm["class_prob"] = class_prob
            
        adata.obsm[self.params.latent_representation_cell] = emb_cell
        adata.obsm[self.params.latent_representation_niche] = emb_niche
        
        return adata
