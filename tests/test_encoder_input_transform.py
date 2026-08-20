from types import SimpleNamespace

import torch
from torch import nn

from gsMap.find_latent.gnn import train_step
from gsMap.find_latent.gnn.encoder_decoder import Encoder, EncoderInputTransform
from gsMap.find_latent.gnn.st_model import StEmbeding


class _CaptureTransform(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input = None

    def forward(self, x, batch):
        self.input = x.detach().clone()
        return torch.zeros((x.shape[0], self.hidden_size), dtype=x.dtype)


def _encoder(
    *, input_transform: EncoderInputTransform
) -> tuple[Encoder, _CaptureTransform]:
    encoder = Encoder(
        input_size=2,
        hidden_size=4,
        emb_size=2,
        batch_emb_size=1,
        module_dim=2,
        hidden_gmf=2,
        n_modules=1,
        nhead=1,
        n_enc_layer=1,
        use_tf=False,
        variational=False,
        input_transform=input_transform,
    )
    capture = _CaptureTransform(hidden_size=4)
    encoder.tf = capture
    encoder.mlp = nn.Identity()
    return encoder, capture


def test_count_encoder_receives_log1p_input():
    encoder, capture = _encoder(input_transform=EncoderInputTransform.LOG1P)
    raw_counts = torch.tensor([[0.0, 3.0], [8.0, 15.0]])

    encoder(raw_counts, torch.zeros((2, 1)))

    torch.testing.assert_close(capture.input, torch.log1p(raw_counts))


def test_non_count_encoder_preserves_input():
    encoder, capture = _encoder(input_transform=EncoderInputTransform.IDENTITY)
    normalized = torch.tensor([[-1.0, 0.5], [0.0, 2.0]])

    encoder(normalized, torch.zeros((2, 1)))

    torch.testing.assert_close(capture.input, normalized)


def _model(distribution: str, batch_label_classes=None, variational=False) -> StEmbeding:
    batch_size = len(batch_label_classes) if batch_label_classes is not None else 1
    return StEmbeding(
        input_size=[2, 4],
        hidden_size=4,
        embedding_size=2,
        batch_embedding_size=2,
        out_put_size=2,
        batch_size=batch_size,
        class_size=1,
        distribution=distribution,
        module_dim=2,
        hidden_gmf=2,
        n_modules=1,
        nhead=1,
        n_enc_layer=1,
        use_tf=False,
        variational=variational,
        batch_label_classes=batch_label_classes,
    )


def test_count_model_enables_log1p_for_both_encoders():
    model = _model("nb")

    assert [encoder.input_transform for encoder in model.encoder] == [
        EncoderInputTransform.LOG1P,
        EncoderInputTransform.LOG1P,
    ]


def test_count_model_applies_log1p_to_both_encoder_inputs():
    model = _model("nb")
    model.eval()
    raw_counts = torch.tensor([[0.0, 3.0], [8.0, 15.0]])
    graph_counts = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    captures = []
    for encoder in model.encoder:
        capture = _CaptureTransform(hidden_size=4)
        encoder.tf = capture
        captures.append(capture)

    model.encode([raw_counts, graph_counts], torch.zeros(2, dtype=torch.long))

    torch.testing.assert_close(captures[0].input, torch.log1p(raw_counts))
    torch.testing.assert_close(captures[1].input, torch.log1p(graph_counts))


def test_gaussian_model_does_not_log_transform_encoder_inputs():
    model = _model("gaussian")

    assert [encoder.input_transform for encoder in model.encoder] == [
        EncoderInputTransform.IDENTITY,
        EncoderInputTransform.IDENTITY,
    ]


def test_old_checkpoint_without_input_transform_version_is_incompatible():
    model = _model("nb")
    old_state = {
        name: value
        for name, value in model.state_dict().items()
        if name != "_encoder_input_transform_version"
    }

    assert not model.has_compatible_encoder_input_transform(old_state)
    assert model.has_compatible_encoder_input_transform(model.state_dict())


def test_old_identity_checkpoint_remains_compatible():
    model = _model("gaussian")
    old_state = {
        name: value
        for name, value in model.state_dict().items()
        if name != "_encoder_input_transform_version"
    }

    assert model.has_compatible_encoder_input_transform(old_state)
    assert model.load_compatible_state_dict(old_state)


def test_checkpoint_requires_matching_batch_label_encoding():
    model = _model("nb", batch_label_classes=["sample-a", "sample-b"])
    matching_state = model.state_dict()
    old_state = {
        name: value
        for name, value in matching_state.items()
        if name != "_batch_label_encoding_signature"
    }
    reversed_model = _model("nb", batch_label_classes=["sample-b", "sample-a"])

    assert model.has_compatible_batch_label_encoding(matching_state)
    assert not model.has_compatible_batch_label_encoding(old_state)
    assert not reversed_model.has_compatible_batch_label_encoding(matching_state)


def test_reconstruction_loss_keeps_raw_count_target(monkeypatch):
    raw_counts = torch.tensor([[0.0, 3.0], [8.0, 15.0]])
    observed_targets = []

    def capture_target(x_hat, target, logtheta, zi_logit, distribution):
        observed_targets.append(target)
        return torch.tensor(0.0)

    monkeypatch.setattr(train_step, "rec_loss", capture_target)
    trainer = SimpleNamespace(
        mode="reconstruction",
        distribution="nb",
        model=SimpleNamespace(encoder=[SimpleNamespace(kl_loss=lambda: 0.0)]),
    )

    train_step.ModelTrain.compute_loss(
        trainer,
        x_hat=[torch.ones_like(raw_counts)],
        x=raw_counts,
        log_theata=torch.zeros_like(raw_counts),
        zi_logit=[torch.zeros_like(raw_counts)],
        x_class=None,
        labels=None,
    )

    assert observed_targets[0] is raw_counts


def test_final_embedding_uses_variational_posterior_mean():
    model = _model("nb", variational=True)
    model.eval()
    raw_counts = torch.tensor([[0.0, 3.0], [8.0, 15.0]])
    graph_counts = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    batches = torch.zeros(2, dtype=torch.long)
    batch_embedding = model._process_batch(batches)

    observed = model.encode([raw_counts, graph_counts], batches)
    expected = [
        encoder.encode_mean(counts, batch_embedding)
        for encoder, counts in zip(model.encoder, [raw_counts, graph_counts], strict=True)
    ]

    for observed_embedding, expected_mean in zip(observed, expected, strict=True):
        torch.testing.assert_close(observed_embedding, expected_mean)
    assert model.latent_embedding_statistic == "posterior_mean"


def test_non_variational_embedding_metadata_uses_deterministic_label():
    model = _model("gaussian", variational=False)

    assert model.latent_embedding_statistic == "deterministic_encoder_output"
