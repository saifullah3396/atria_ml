"""
Sequence ANLS Metric Module

This module provides utilities for computing the Average Normalized Levenshtein Similarity (ANLS)
metric for sequence-based question answering tasks. It includes functions for postprocessing
predictions, computing ANLS scores, and defining a sequence ANLS metric.

Functions:
    - convert_to_list: Converts input to a list.
    - postprocess_qa_predictions: Postprocesses predictions for question answering tasks.
    - anls_metric_str: Computes ANLS scores for predictions and gold labels.
    - sequence_anls: Defines a sequence ANLS metric for use in training.

Dependencies:
    - numpy: For numerical operations.
    - textdistance: For computing Levenshtein distance.
    - torch: For PyTorch operations.
    - anls: For ANLS score computation.
    - core.logger: For logging utilities.
    - core.metrics.common.epoch_dict_metric: For defining epoch-level metrics.

Author: Your Name (your.email@example.com)
Date: 2025-04-14
Version: 1.0.0
License: MIT
"""

import collections
import json
from collections.abc import Callable

import numpy as np
import textdistance as td
import torch
from anls import anls_score
from ignite.metrics import Metric

from atria_ml.registry import METRIC
from atria_ml.training.metrics.common.epoch_dict_metric import EpochDictMetric
from atria_ml.training.metrics.qa.output_transforms import (
    _sequence_anls_output_transform,
)


def convert_to_list(x):
    """
    Converts the input to a list.

    Args:
        x: Input data, which can be a list or a torch.Tensor.

    Returns:
        A list representation of the input.
    """
    if isinstance(x, list):
        return x
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    return x


def postprocess_qa_predictions(
    words,
    word_ids,
    sequence_ids,
    question_ids,
    start_logits,
    end_logits,
    n_best_size: int = 20,
    max_answer_length: int = 100,
):
    """
    Postprocesses predictions for question answering tasks.

    Args:
        words: List of words in the context.
        word_ids: Word IDs corresponding to the context.
        sequence_ids: Sequence IDs for the input.
        question_ids: IDs for the questions.
        start_logits: Start logits from the model.
        end_logits: End logits from the model.
        n_best_size: Number of best predictions to consider.
        max_answer_length: Maximum length of the answer.

    Returns:
        A tuple containing:
        - all_predictions: Ordered dictionary of predictions.
        - all_predictions_list: List of predictions with question IDs.
    """
    word_ids = convert_to_list(word_ids)
    sequence_ids = convert_to_list(sequence_ids)
    question_ids = convert_to_list(question_ids)

    features_per_example = collections.defaultdict(list)
    for feature_id, question_id in enumerate(
        question_ids
    ):  # each example has a unique question id
        features_per_example[question_id].append(feature_id)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_predictions_list = []
    all_nbest_json = collections.OrderedDict()

    # Let's loop over all the examples!
    for question_id, feature_indices in features_per_example.items():
        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            feature_start_logits = start_logits[feature_index].numpy()
            feature_end_logits = end_logits[feature_index].numpy()

            feature_word_ids = word_ids[feature_index]
            feature_sequence_ids = sequence_ids[feature_index]

            num_question_tokens = 0
            while feature_sequence_ids[num_question_tokens] != 1:
                num_question_tokens += 1

            feature_null_score = feature_start_logits[0] + feature_end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": feature_start_logits[0],
                    "end_logit": feature_end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(feature_start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(feature_end_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index < num_question_tokens
                        or end_index < num_question_tokens
                        or start_index >= len(feature_word_ids)
                        or end_index >= len(feature_word_ids)
                        or feature_word_ids[start_index] is None
                        or feature_word_ids[end_index] is None
                    ):
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    prelim_predictions.append(
                        {
                            "word_ids": (
                                feature_word_ids[start_index],
                                feature_word_ids[end_index],
                            ),
                            "score": feature_start_logits[start_index]
                            + feature_end_logits[end_index],
                            "start_logit": feature_start_logits[start_index],
                            "end_logit": feature_end_logits[end_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        first_feature_id = features_per_example[question_id][0]
        context = words[first_feature_id]

        for pred in predictions:
            offsets = pred.pop("word_ids")
            pred["text"] = " ".join(
                [x.strip() for x in context[offsets[0] : offsets[1] + 1]]
            )

        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):
            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        for prob, pred in zip(probs, predictions, strict=True):
            pred["probability"] = prob

        all_predictions[question_ids[feature_index]] = predictions[0]["text"]
        all_predictions_list.append(
            {
                "questionId": question_ids[feature_index],
                "answer": predictions[0]["text"],
            }
        )
        all_nbest_json[question_ids[feature_index]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, np.float16 | np.float32 | np.float64)
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    with open("results.json", "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    return all_predictions, all_predictions_list


def anls_metric_str(
    predictions: list[list[str]], gold_labels: list[list[str]], tau=0.5, rank=0
):
    """
    Computes ANLS scores for predictions and gold labels.

    Args:
        predictions: List of predicted answers.
        gold_labels: List of gold answers, each instance may have multiple gold labels.
        tau: Threshold for normalized Levenshtein similarity.
        rank: Rank of the computation.

    Returns:
        A tuple containing:
        - res: List of ANLS scores for each instance.
        - Average ANLS score across all instances.
    """
    res = []
    for _, (preds, golds) in enumerate(zip(predictions, gold_labels, strict=True)):
        max_s = 0
        for pred in preds:
            for gold in golds:
                dis = td.levenshtein.distance(pred.lower(), gold.lower())
                max_len = max(len(pred), len(gold))
                if max_len == 0:
                    s = 0
                else:
                    nl = dis / max_len
                    s = 1 - nl if nl < tau else 0
                max_s = max(s, max_s)
        res.append(max_s)
    return res, sum(res) / len(res)


@METRIC.register("sequence_anls", output_transform=_sequence_anls_output_transform)
def sequence_anls(
    output_transform: Callable, device: str | torch.device, threshold: float = 0.5
) -> Metric:
    """
    Defines a sequence ANLS metric for use in training.

    Args:
        output_transform: Function to transform the output.
        device: Device to perform computations on.
        threshold: Threshold for ANLS score computation.

    Returns:
        An instance of EpochDictMetric for computing sequence ANLS.
    """

    def wrap(
        words, word_ids, sequence_ids, question_ids, start_logits, end_logits, answers
    ):
        all_predictions, all_predictions_list = postprocess_qa_predictions(
            words=words,
            word_ids=word_ids.detach().cpu(),
            sequence_ids=sequence_ids.detach().cpu(),
            question_ids=question_ids.detach().cpu(),
            start_logits=start_logits.detach().cpu(),
            end_logits=end_logits.detach().cpu(),
        )
        print(f"Total predictions: {len(all_predictions_list)}")

        true_answers_per_example = collections.defaultdict(list)
        question_ids = convert_to_list(question_ids)
        for question_id, answer in zip(question_ids, answers, strict=True):
            true_answers_per_example[question_id].append(answer)
        true_answers_per_example = {
            k: v[0] for k, v in true_answers_per_example.items()
        }
        true_answers_per_example = list(true_answers_per_example.values())

        all_pred_answers = [prediction["answer"] for prediction in all_predictions_list]
        assert len(all_pred_answers) == len(true_answers_per_example), (
            f"The number of predicted answers must match the lists of ground truth answers."
            f"len(all_pred_answers){len(all_pred_answers)} != len(true_answers_per_example)({len(true_answers_per_example)})"
        )
        print("Length of all_pred_answers:", len(all_pred_answers))
        print("Length of true_answers_per_example:", len(true_answers_per_example))
        print(f"prediction: {all_pred_answers[:20]}")
        print(f"gold_answers: {true_answers_per_example[:20]}")
        anls_scores = [
            anls_score(pred, target, threshold=threshold)
            for pred, target in zip(
                all_pred_answers, true_answers_per_example, strict=True
            )
        ]
        anls = sum(anls_scores) / len(anls_scores)
        return anls

    return EpochDictMetric(wrap, output_transform=output_transform, device=device)
