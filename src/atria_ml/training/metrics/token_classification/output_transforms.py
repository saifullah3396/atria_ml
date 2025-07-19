from atria_models.data_types.outputs import TokenClassificationModelOutput


def _output_transform(model_output: TokenClassificationModelOutput):
    assert isinstance(model_output, TokenClassificationModelOutput), (
        f"Expected {TokenClassificationModelOutput}, got {type(model_output)}"
    )
    return model_output.target_labels, model_output.predicted_labels
