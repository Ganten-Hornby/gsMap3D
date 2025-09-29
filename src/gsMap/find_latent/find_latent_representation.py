import os
import torch
import numpy as np
import logging
import random
import yaml
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import (
    DataLoader,
    random_split,
    TensorDataset,
    SubsetRandomSampler,
)
from collections import OrderedDict

from .GNN.train_step import ModelTrain
from .GNN.STmodel import StEmbeding
from .ST_process import TrainingData, find_common_hvg, create_subsampled_adata, InferenceData
from ..config import FindLatentRepresentationsConfig

from operator import itemgetter

logger = logging.getLogger(__name__)



def set_seed(seed_value):
    """
    Set seed for reproducibility in PyTorch and other libraries.
    """
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        logger.info("Using GPU for computations.")
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    else:
        logger.info("Using CPU for computations.")


def index_splitter(n, splits):
    idx = torch.arange(n)
    splits_tensor = torch.as_tensor(splits)
    multiplier = n / splits_tensor.sum()
    splits_tensor = (multiplier * splits_tensor).long()
    diff = n - splits_tensor.sum()
    splits_tensor[0] += diff
    return random_split(idx, splits_tensor)


def run_find_latent_representation(config: FindLatentRepresentationsConfig) -> Dict[str, Any]:
    """
    Run the find latent representation pipeline.
    
    Args:
        config: FindLatentRepresentationsConfig object with all necessary parameters
        
    Returns:
        Dictionary containing metadata about the run including config, model info,
        training info, outputs, and annotation info
    """
    logger.info(f'Project dir: {config.project_dir}')
    set_seed(2024)

    # Find the hvg
    hvg, n_cell_used, gene_name_dict = find_common_hvg(config.sample_h5ad_dict, config)
    common_genes = np.array(list(gene_name_dict.keys()))

    # Create subsampled concatenated adata with sample-specific stratified sampling
    training_adata = create_subsampled_adata(config.sample_h5ad_dict, n_cell_used, config)

    # Prepare the trainning data
    get_trainning_data = TrainingData(config)
    get_trainning_data.prepare(training_adata, hvg)

    # Configure the distribution
    if config.data_layer in ["count", "counts"]:
        distribution = config.distribution
        variational = True
        use_tf = config.use_tf
    else:
        distribution = "gaussian"
        variational = False
        use_tf = False

    # Instantiation the LGCN VAE
    input_size = [
        get_trainning_data.expression_merge.size(1),
        get_trainning_data.expression_gcn_merge.size(1),
    ]
    class_size = len(torch.unique(get_trainning_data.label_merge))
    batch_size = get_trainning_data.batch_size
    cell_size, out_size = get_trainning_data.expression_merge.shape
    label_name = get_trainning_data.label_name

    # Configure the batch embedding dim
    batch_embedding_size = 64

    # Configure the model
    gsmap_lgcn_model = StEmbeding(
        # parameter of VAE
        input_size=input_size,
        hidden_size=config.hidden_size,
        embedding_size=config.embedding_size,
        batch_embedding_size=batch_embedding_size,
        out_put_size=out_size,
        batch_size=batch_size,
        class_size=class_size,
        # parameter of transformer
        module_dim=config.module_dim,
        hidden_gmf=config.hidden_gmf,
        n_modules=config.n_modules,
        nhead=config.nhead,
        n_enc_layer=config.n_enc_layer,
        # parameter of model structure
        distribution=distribution,
        use_tf=use_tf,
        variational=variational,
    )

    # Configure the optimizer
    optimizer = torch.optim.Adam(gsmap_lgcn_model.parameters(), lr=1e-3)
    logger.info(
        f"gsMap-LGCN parameters: {sum(p.numel() for p in gsmap_lgcn_model.parameters())}."
    )
    logger.info(f"Number of cells used in trainning: {cell_size}.")

    # Split the data to trainning (80%) and validation (20%).
    train_idx, val_idx = index_splitter(
        get_trainning_data.expression_gcn_merge.size(0), [80, 20]
    )
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Configure the data loader
    dataset = TensorDataset(
        get_trainning_data.expression_gcn_merge,
        get_trainning_data.batch_merge,
        get_trainning_data.expression_merge,
        get_trainning_data.label_merge,
    )
    train_loader = DataLoader(
        dataset=dataset, batch_size=config.batch_size, sampler=train_sampler
    )
    val_loader = DataLoader(
        dataset=dataset, batch_size=config.batch_size, sampler=val_sampler
    )

    # Model trainning
    gsMap_embedding_finder = ModelTrain(
        gsmap_lgcn_model,
        optimizer,
        distribution,
        mode="reconstruction",
        lr=1e-3,
        model_path=config.model_path,
    )
    gsMap_embedding_finder.set_loaders(train_loader, val_loader)
    print(gsMap_embedding_finder.model)

    if not os.path.exists(config.model_path):
        # reconstruction
        gsMap_embedding_finder.train(config.itermax, patience=config.patience)

        # classification
        if config.two_stage and config.annotation is not None:
            gsMap_embedding_finder.model.load_state_dict(torch.load(config.model_path))
            gsMap_embedding_finder.mode = "classification"
            gsMap_embedding_finder.train(config.itermax, patience=config.patience)
    else:
        logger.info(f"Model found at {config.model_path}. Skipping training.")

    # Load the best model
    gsMap_embedding_finder.model.load_state_dict(torch.load(config.model_path))
    gsmap_embedding_model = gsMap_embedding_finder.model

    # Configure the inference
    infer = InferenceData(hvg, batch_size, gsmap_embedding_model, label_name, config)


    output_h5ad_path_dict = OrderedDict(
        {sample_name: config.latent_dir / f"{sample_name}_add_latent.h5ad"
            for sample_name in config.sample_h5ad_dict.keys()}
    )
    for st_id, (sample_name, st_file) in enumerate(config.sample_h5ad_dict.items()):

        output_path = output_h5ad_path_dict[sample_name]

        # Infer the embedding
        adata = infer.infer_embedding_single(st_id, st_file)

        # Transfer the gene name
        common_genes = np.array(list(gene_name_dict.keys()))
        common_genes_transfer = np.array(itemgetter(*common_genes)(gene_name_dict))
        adata = adata[:, common_genes].copy()
        adata.var_names = common_genes_transfer


        # Compute the depth
        if config.data_layer in ["count", "counts"]:
            adata.obs['depth'] = np.array(adata.layers[config.data_layer].sum(axis=1)).flatten()

        # Save the ST data with embeddings
        adata.write_h5ad(output_path)

        logger.info(f"Saved latent representation to {output_path}")

    # Convert config to dict with all Path objects as strings
    config_dict = config.to_dict_with_paths_as_strings()
    
    # Convert output_h5ad_path_dict to strings
    output_h5ad_path_dict_str = {k: str(v) for k, v in output_h5ad_path_dict.items()}
    
    # Save metadata
    metadata = {
        "config": config_dict,
        "model_info": {
            "model_path": str(config.model_path),
            "n_parameters": int(sum(p.numel() for p in gsmap_embedding_model.parameters())),
            "input_size": [int(x) for x in input_size],
            "hidden_size": int(config.hidden_size),
            "embedding_size": int(config.embedding_size),
            "batch_embedding_size": int(batch_embedding_size),
            "class_size": int(class_size),
            "distribution": distribution,
            "variational": variational,
            "use_tf": use_tf
        },
        "training_info": {
            "n_cells_used": int(cell_size),
            "n_genes_used": int(len(hvg)),
            "n_common_genes": int(len(common_genes)),
            "batch_size": int(config.batch_size),
            "n_epochs": int(config.itermax),
            "patience": int(config.patience),
            "two_stage": config.two_stage
        },
        "outputs": {
            "latent_files": output_h5ad_path_dict_str,
            "n_sections": len(config.sample_h5ad_dict)
        },
        "annotation_info": {
            "annotation_key": config.annotation,
            "n_classes": int(class_size),
            "label_names": label_name if isinstance(label_name, list) else label_name.tolist() if hasattr(label_name, 'tolist') else list(label_name),
        }
    }
    
    # Save metadata to YAML file
    metadata_path = config.find_latent_metadata_path
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved metadata to {metadata_path}")
    
    return metadata
