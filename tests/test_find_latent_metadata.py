from gsMap.find_latent.find_latent_representation import _serialize_label_names


def test_serialize_label_names_without_annotation() -> None:
    assert _serialize_label_names(None) == []


def test_serialize_label_names_from_array_like() -> None:
    class ArrayLikeLabels:
        def tolist(self) -> list[str]:
            return ["left", "right"]

    assert _serialize_label_names(ArrayLikeLabels()) == ["left", "right"]

