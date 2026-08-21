import torch

from gsMap.find_latent.gnn.st_model import StEmbeding


def _make_variational_model():
    return StEmbeding(
        input_size=[4, 8],
        hidden_size=32,
        embedding_size=8,
        batch_embedding_size=16,
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


def test_explicit_means_are_deterministic_while_default_inference_samples_z():
    torch.manual_seed(42)
    model = _make_variational_model().eval()
    expression = torch.rand(6, 4)
    spatial_expression = torch.rand(6, 8)

    with torch.no_grad():
        first_mean = model.encode([expression, spatial_expression], sample=False)
        second_mean = model.encode([expression, spatial_expression], sample=False)
        first_sample = model.encode([expression, spatial_expression])
        second_sample = model.encode([expression, spatial_expression])

    for first, second in zip(first_mean, second_mean, strict=True):
        torch.testing.assert_close(first, second)
    assert any(
        not torch.equal(first, second)
        for first, second in zip(first_sample, second_sample, strict=True)
    )


def test_two_layer_batch_conditioned_decoder_has_expected_connections():
    model = _make_variational_model().eval()

    encoder_input = model.encoder[0].tf.transform[0]
    decoder = model.decoder["reconstruction"]
    decoder_input = decoder.mlp[0][0]
    decoder_second_layer = decoder.mlp[1][0]

    assert encoder_input.in_features == 4
    assert model.batch_embedding.embedding_dim == 16
    assert decoder_input.in_features == 8 + 16
    assert decoder_second_layer.in_features == 32 + 16
    assert decoder_second_layer.out_features == 32
    assert decoder.fc_rec.in_features == 32
    assert decoder.zi_logit.in_features == 32


def test_encoder_is_invariant_to_batch_embedding_parameters():
    torch.manual_seed(42)
    model = StEmbeding(
        input_size=[4],
        hidden_size=32,
        embedding_size=8,
        batch_embedding_size=16,
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
    ).eval()
    expression = torch.rand(6, 4)

    with torch.no_grad():
        before = model.encode([expression], sample=False)[0]
        model.batch_embedding.weight.normal_(mean=100.0, std=20.0)
        after = model.encode([expression], sample=False)[0]

    torch.testing.assert_close(before, after)
