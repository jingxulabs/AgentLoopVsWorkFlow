"""Official HotpotQA answer metrics.

Implemented to match hotpot_evaluate_v1.py from the HotpotQA reference repo
(https://github.com/hotpotqa/hotpot, Apache-2.0). normalize_answer is the
standard SQuAD normalisation (lowercase -> strip punctuation -> strip articles
-> collapse whitespace); f1_score adds HotpotQA's span-free short circuit so a
predicted "yes" scores 0 against a gold span rather than partial-crediting on
token overlap.

We score answer EM/F1 only, not supporting-fact EM/F1. The arms differ in
control flow, and only the workflow arm has a well-defined notion of "the
paragraphs I committed to" - scoring Sup would compare two different things.
"""

from __future__ import annotations

import re
import string
from collections import Counter

_SPAN_FREE = ("yes", "no", "noanswer")


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Returns (f1, precision, recall)."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero_metric = (0.0, 0.0, 0.0)

    # A yes/no/noanswer answer is correct only when it matches exactly; token
    # overlap must not earn partial credit against a span.
    if normalized_prediction in _SPAN_FREE and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in _SPAN_FREE and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def score_answer(prediction: str, ground_truth: str) -> dict:
    em = exact_match_score(prediction, ground_truth)
    f1, precision, recall = f1_score(prediction, ground_truth)
    return {
        "em": 1.0 if em else 0.0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
