from atria_models.data_types.outputs import ClassificationModelOutput


def _output_transform(output: ClassificationModelOutput):
    assert isinstance(output, ClassificationModelOutput), (
        f"Expected {ClassificationModelOutput}, got {type(output)}"
    )
    return output.logits, output.label.value
