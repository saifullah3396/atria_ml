from atria_models.data_types.outputs import LayoutTokenClassificationModelOutput


def _output_transform(output: LayoutTokenClassificationModelOutput):
    assert isinstance(output, LayoutTokenClassificationModelOutput), (
        f"Expected {LayoutTokenClassificationModelOutput}, got {type(output)}"
    )

    # y_pred, y, y_bbox = output.logits, output.token_labels, output.token_bboxes
    # pred corresponds to logits instead of argmax outputs
    return output.logits, output.token_labels, output.token_bboxes
