"""Regression test for the exact kind of drift this project has repeatedly
had to hand-diagnose: group_rating() is reimplemented once per model (plus
shared_baseline.py's own canonical copy) and every copy's docstring says
"must stay in sync" with the others, but nothing previously enforced that
automatically. This test imports every implementation and checks they agree
on the same 4-tier bucket for every raw letter grade.

Deliberately NOT testing case/whitespace normalisation here: shared_baseline
and xgboost_stuff/preprocessing.py normalise input (`.strip().upper()`)
before matching, but predict_decision_tree.py and predict_logistic_regression
.py do not -- a real, separate gap, not something to paper over in this test.
Every implementation agrees on clean, upper-case grades (what the actual
dataset contains), which is what's checked here.
"""

import shared_baseline
import preprocessing as xgboost_preprocessing
import predict_decision_tree
import predict_random_forest
import predict_logistic_regression

RAW_GRADES = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D", "NR"]

IMPLEMENTATIONS = {
    "shared_baseline": shared_baseline.group_rating,
    "xgboost_preprocessing": xgboost_preprocessing.group_rating,
    "decision_tree": predict_decision_tree.group_rating,
    "random_forest": predict_random_forest.group_rating,
    "logistic_regression": predict_logistic_regression.group_rating,
}


def _normalise(label):
    """Collapse the hyphen/underscore formatting difference (documented,
    intentional) so only the *bucket* is compared, not the string style.
    """
    return str(label).replace("_", "-")


def test_all_implementations_agree_on_every_grade():
    for grade in RAW_GRADES:
        results = {
            name: _normalise(fn(grade)) for name, fn in IMPLEMENTATIONS.items()
        }
        distinct = set(results.values())
        assert len(distinct) == 1, (
            f"group_rating() implementations disagree for grade {grade!r}: {results}"
        )


def test_expected_tiers():
    expected = {
        "AAA": "Investment-High", "AA": "Investment-High", "A": "Investment-High",
        "BBB": "Investment-Low",
        "BB": "Speculative", "B": "Speculative",
        "CCC": "Distressed", "CC": "Distressed", "C": "Distressed", "D": "Distressed",
        "NR": "Unknown",
    }
    for grade, tier in expected.items():
        assert _normalise(shared_baseline.group_rating(grade)) == tier
