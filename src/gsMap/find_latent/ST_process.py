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
from rich.progress import track
from scipy.special import softmax
from .GNN.GCN import GCN, build_spatial_graph
from gsMap.config import FindLatentRepresentationsConfig

logger = logging.getLogger(__name__)

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

    for sample_name, st_file in track(sample_h5ad_dict.items(), description="Finding HVGs"):
        adata_temp = sc.read_h5ad(st_file)
        # sc.pp.filter_genes(adata_temp, min_counts=1)

        # Filter out mitochondrial and hemoglobin genes
        gene_keep = ~adata_temp.var_names.str.match(re.compile(r'^(HB.-|MT-)', re.IGNORECASE))
        adata_temp = adata_temp[:,gene_keep].copy()

        # Set data layer
        # print(params.data_layer)
        if params.data_layer not in adata_temp.layers:
            if adata_temp.X is not None and np.issubdtype(
                adata_temp.X.dtype,
                np.integer
            ):
                logger.info(
                    f'Data layer {params.data_layer} not found or not integer'
                    f', falling back to adata.X'
                )
                adata_temp.layers[params.data_layer] = adata_temp.X.copy()
                params.data_layer = 'count'
            else:
                params.data_layer = None
        else:
            adata_temp.X = adata_temp.layers[params.data_layer]

        # Identify highly variable genes
        flavor = "seurat_v3" if params.data_layer in ["count", "counts", "impute_count"] else "seurat"
        sc.pp.highly_variable_genes(
            adata_temp, n_top_genes=params.feat_cell, subset=False, flavor=flavor
        )
        var_df = adata_temp.var
        var_df["gene"] = var_df.index.tolist()
        variances_list.append(var_df)

        cell_number.append(adata_temp.n_obs)

    # Find the common genes across all datasets
    common_genes = np.array(
        list(set.intersection(
            *map(set, [st.index.to_list() for st in variances_list])))
    )

    # Aggregate variances and identify HVGs
    df = pd.concat(variances_list, axis=0)
    df["highly_variable"] = df["highly_variable"].astype(int)
    if flavor=='seurat_v3':
        df = df.groupby("gene", observed=True).agg(
            dict(
                variances_norm="median",
                highly_variable="sum",
            )
        )
        df = df.loc[common_genes]
        df["highly_variable_nbatches"] = df["highly_variable"]
        df.sort_values(
            ["highly_variable_nbatches", "variances_norm"],
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
        gene_name_dict = dict(zip(common_genes,homologs.loc[common_genes].HUMAN_GENE_SYM.values))
    else:
        gene_name_dict = dict(zip(common_genes,common_genes))
    return hvg, n_cell_used, gene_name_dict


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

        # Set data layers
        if params.data_layer not in adata.layers:
            if adata.X is not None and np.issubdtype(adata.X.dtype, np.integer):
                logger.info(
                    f'Data layer {params.data_layer} not found, falling back to adata.X'
                )
                adata.layers[params.data_layer] = adata.X.copy()
                params.data_layer = 'count'
            else:
                params.data_layer = None
        else:
            adata.X = adata.layers[params.data_layer]

        # Filter cells based on annotation if provided
        if params.annotation is not None:
            adata = adata[~adata.obs[params.annotation].isnull()]

        # Perform stratified sampling within this sample
        if params.do_sampling:
            if params.annotation is None:
                # Simple random sampling
                num_cell = min(adata.n_obs, n_cell_used[st_id])
                logger.info(f"Downsampling {sample_name} to {num_cell} cells...")
                random_indices = np.random.choice(adata.n_obs, num_cell, replace=False)
                adata = adata[random_indices].copy()
            else:
                # Stratified sampling based on sample-specific annotation distribution
                sample_annotation_counts = adata.obs[params.annotation].value_counts()
                sample_total_cells = len(adata)
                target_total_cells = min(sample_total_cells, n_cell_used[st_id])

                # Calculate sample-specific annotation proportions
                sample_annotation_proportions = sample_annotation_counts / sample_total_cells

                # Calculate target cells for each annotation in this sample
                target_cells_per_annotation = (sample_annotation_proportions * target_total_cells).astype(int)

                logger.info(f"Downsampling {sample_name} to {target_total_cells} cells...")
                logger.info("---Sample-specific annotation distribution-----")
                for ann, count in target_cells_per_annotation.items():
                    logger.info(f"{ann}: {count} cells")

                # Perform stratified sampling
                sampled_cells = (
                    adata.obs.groupby(params.annotation, group_keys=False)
                    .apply(
                        lambda x: x.sample(
                            max(min(target_cells_per_annotation.get(x.name, 0), len(x)), 1),
                            replace=False,
                        )
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
        logger.info(f"Subsampled {sample_name}: {adata.n_obs} cells, {adata.n_vars} genes")

    # Concatenate all samples
    logger.info("Concatenating all subsampled data...")
    concatenated_adata = sc.concat(subsampled_adatas, axis=0, join='inner',
                                   index_unique='_', fill_value=0)

    logger.info(f"Final concatenated adata: {concatenated_adata.n_obs} cells, {concatenated_adata.n_vars} genes")

    return concatenated_adata


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
        
        # Set data layers
        if not hasattr(adata, 'layers') or self.params.data_layer not in adata.layers:
            if adata.X is not None and np.issubdtype(adata.X.dtype, np.integer):
                logger.info(
                    f'Data layer {self.params.data_layer} not found in layers or layers missing, '
                    f'falling back to adata.X'
                )
                adata.X = adata.X  # Use adata.X directly
        else:
            adata.X = adata.layers[self.params.data_layer]
        
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
        emb, emb_gcn, class_prob = [], [], []

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
                emb.append(mu_focal[0].cpu().numpy())
                emb_gcn.append(mu_focal[1].cpu().numpy())

        # Concatenate results and store embeddings in adata
        emb = np.concatenate(emb, axis=0)
        emb_gcn = np.concatenate(emb_gcn, axis=0)
        class_prob = np.concatenate(class_prob, axis=0)
        
        # if self.label_name is not None:
        #     class_prob = pd.DataFrame(softmax(class_prob,axis=1), columns=self.label_name,index=adata.obs_names)
        #     adata.obsm["class_prob"] = class_prob
            
        adata.obsm["emb"] = emb
        adata.obsm["emb_gcn"] = emb_gcn
        
        return adata
