import pytest
import torch
from torch import nn

from gsMap.find_latent.gnn.encoder_decoder import Encoder
from gsMap.find_latent.gnn.gene_former import (
    GeneModuleFormer,
    GeneModuler,
    PositionalEncoding,
)


class _ZeroNorm(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def test_gene_moduler_uses_normalized_expression():
    moduler = GeneModuler(input_dim=3, hidden_dim=2, n_modules=2)
    moduler.layernorm = _ZeroNorm()
    with torch.no_grad():
        moduler.extractor.weights.fill_(1.0)

    result = moduler(torch.ones(1, 3))

    torch.testing.assert_close(result, torch.zeros_like(result))


def test_positional_encoding_uses_each_position_and_supports_odd_dimensions():
    positional_encoding = PositionalEncoding(d_model=5, max_len=4)

    result = positional_encoding(torch.zeros(2, 4, 5))

    assert result.shape == (2, 4, 5)
    assert not torch.equal(result[:, 0, :], result[:, 1, :])


def test_transformer_hidden_dimension_must_be_divisible_by_attention_heads():
    with pytest.raises(ValueError, match="must be divisible"):
        GeneModuleFormer(hidden_dim=30, nhead=4)


def test_transformer_encoder_returns_embedding_without_sampling():
    encoder = Encoder(
        input_size=4,
        hidden_size=32,
        emb_size=8,
        module_dim=5,
        hidden_gmf=32,
        n_modules=4,
        nhead=4,
        n_enc_layer=1,
        use_tf=True,
        variational=True,
    ).eval()
    expression = torch.rand(6, 4)

    with torch.no_grad():
        result = encoder(expression, sample=False)

    assert result.shape == (6, 8)
    assert torch.isfinite(result).all()
