import torch

from gsMap.find_latent.gnn.st_model import StEmbeding


def _make_variational_model():
    return StEmbeding(
        input_size=[4, 8],
        hidden_size=32,
        embedding_size=8,
        batch_embedding_size=4,
        out_put_size=4,
        batch_size=2,
        class_size=2,
        distribution="nb",
        module_dim=4,
        hidden_gmf=32,
        n_modules=4,
        nhead=4,
        n_enc_layer=1,
        use_tf=False,
        variational=True,
    )


def test_mean_embeddings_are_deterministic_while_training_interface_still_samples():
    torch.manual_seed(42)
    model = _make_variational_model().eval()
    expression = torch.rand(6, 4)
    spatial_expression = torch.rand(6, 8)
    batches = torch.tensor([0, 0, 0, 1, 1, 1])

    with torch.no_grad():
        first_mean = model.encode([expression, spatial_expression], batches, sample=False)
        second_mean = model.encode([expression, spatial_expression], batches, sample=False)
        first_sample = model.encode([expression, spatial_expression], batches)
        second_sample = model.encode([expression, spatial_expression], batches)

    for first, second in zip(first_mean, second_mean, strict=True):
        torch.testing.assert_close(first, second)
    assert any(
        not torch.equal(first, second)
        for first, second in zip(first_sample, second_sample, strict=True)
    )
