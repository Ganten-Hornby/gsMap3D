import hashlib

import torch
import torch.nn as nn
from torch.nn import functional as F

from .encoder_decoder import Decoder, Encoder


class StEmbeding(nn.Module):
    """
    Learn graph-smoothed and expression embeddings with decoder-only batch correction.

    Args:
        input_size (list): List of input feature sizes for each encoder.
        hidden_size (int): Hidden layer size in encoder/decoder.
        embedding_size (int): Latent embedding size.
        batch_embedding_size (int): Size of batch embedding vector.
        out_put_size (int): Output gene size.
        batch_size (int): Number of batches (not sample count).
        class_size (int): Number of classes for classification.
        distribution (str): Output distribution type ('nb', 'zinb', 'gaussian', etc.).
        Other GNN-related args passed to Encoder.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        embedding_size,
        batch_embedding_size,
        out_put_size,
        batch_size,
        class_size,
        distribution,
        module_dim,
        hidden_gmf,
        n_modules,
        nhead,
        n_enc_layer,
        use_tf=True,
        variational=True,
        batch_representation="embedding",
        dispersion="gene",
        batch_label_classes=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.z_num = len(self.input_size)
        self.distribution = distribution
        self.batch_representation = batch_representation
        self.num_batches = batch_size
        self.latent_embedding_statistic = (
            "posterior_sample" if variational else "deterministic_encoder_output"
        )
        self.register_buffer(
            "_embedding_architecture_version",
            torch.tensor(3, dtype=torch.int8),
        )
        if batch_label_classes is not None:
            encoded_labels = "\0".join(str(label) for label in batch_label_classes).encode()
            signature = hashlib.sha256(encoded_labels).digest()
            self.register_buffer(
                "_batch_label_encoding_signature",
                torch.tensor(list(signature), dtype=torch.uint8),
            )

        self.logtheta = nn.Parameter(torch.randn(batch_size, out_put_size))

        # Handle batch embedding
        if batch_representation == "embedding":
            self.batch_embedding = nn.Embedding(batch_size, batch_embedding_size)
            self.batch_embedding_size = batch_embedding_size
        else:
            self.batch_embedding_size = batch_size  # one-hot case

        # Build encoders for each modality
        self.encoder = nn.ModuleList()
        for eid in range(self.z_num):
            self.encoder.append(
                Encoder(
                    self.input_size[eid],
                    hidden_size,
                    embedding_size,
                    module_dim,
                    hidden_gmf,
                    n_modules,
                    nhead,
                    n_enc_layer,
                    use_tf,
                    variational,
                )
            )

        # Build decoders for reconstruction and classification
        self.decoder = nn.ModuleDict()
        for decoder_type in ["reconstruction", "classification"]:
            self.decoder[decoder_type] = Decoder(
                out_put_size,
                hidden_size,
                embedding_size,
                self.batch_embedding_size,
                class_size,
                decoder_type,
                self.distribution,
            )

    def _process_batch(self, batch):
        if self.batch_representation == "embedding":
            return self.batch_embedding(batch)
        else:
            return F.one_hot(batch, num_classes=self.num_batches).float()

    def has_compatible_batch_label_encoding(self, state_dict) -> bool:
        if not hasattr(self, "_batch_label_encoding_signature"):
            return True
        observed = state_dict.get("_batch_label_encoding_signature")
        if observed is None:
            return False
        expected = self._batch_label_encoding_signature
        return bool(torch.equal(observed.detach().cpu(), expected.detach().cpu()))

    def has_compatible_embedding_architecture(self, state_dict) -> bool:
        observed = state_dict.get("_embedding_architecture_version")
        if observed is None:
            return False
        expected = self._embedding_architecture_version
        return bool(torch.equal(observed.detach().cpu(), expected.detach().cpu()))

    def load_compatible_state_dict(self, state_dict) -> bool:
        if not self.has_compatible_batch_label_encoding(
            state_dict
        ) or not self.has_compatible_embedding_architecture(state_dict):
            return False
        try:
            self.load_state_dict(state_dict)
        except RuntimeError:
            return False
        return True

    def forward(self, x_list, batch):
        batch = self._process_batch(batch)

        if self.distribution in ["nb", "zinb"]:
            library_size = x_list[0].sum(-1, keepdim=True)
        else:
            n = x_list[0].shape[0]
            device = x_list[0].device
            library_size = torch.ones(n, 1, device=device)

        x_rec_list, zi_logit_list, z_list = [], [], []
        for eid in range(self.z_num):
            z = self.encoder[eid](x_list[eid])
            x_rec, zi_logit = self.decoder["reconstruction"](z, batch)
            x_rec = x_rec * library_size

            x_rec_list.append(x_rec)
            zi_logit_list.append(zi_logit)
            z_list.append(z)

        x_class = self._classification(z_list, batch)
        return x_rec_list, x_class, zi_logit_list, z_list

    def _classification(self, z_list, batch):
        z = torch.cat(z_list, dim=1)
        return self.decoder["classification"](z, batch)

    def encode(self, x_list, batch=None, sample=True):
        """Return batch-invariant encoder outputs for downstream embeddings."""

        z_list = []
        for eid in range(self.z_num):
            z = self.encoder[eid](x_list[eid], sample=sample)
            z_list.append(z)
        return z_list
