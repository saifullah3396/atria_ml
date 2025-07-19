import torch
from atria_models.data_types.outputs import QAModelOutput, SequenceQAModelOutput


def _anls_output_transform(output: QAModelOutput) -> tuple[list[str], list[str]]:
    assert isinstance(output, QAModelOutput), (
        f"Expected {QAModelOutput}, got {type(output)}"
    )
    return output.pred_answers, output.target_answers


def _sequence_anls_output_transform(
    output: SequenceQAModelOutput,
) -> tuple[list[str], list[int], list[int], str, torch.Tensor, torch.Tensor, list[str]]:
    assert isinstance(output, SequenceQAModelOutput), (
        f"Expected {SequenceQAModelOutput}, got {type(output)}"
    )
    return (
        output.words,
        output.word_ids,
        output.sequence_ids,
        output.question_id,
        output.start_logits,
        output.end_logits,
        output.gold_answers,
    )
