"""Tests for python_models/shared_baseline.py -- the fair cross-model
comparison pipeline used by all four predict_*.py scripts.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import shared_baseline as sb


def test_clean_dataframe_drops_exact_duplicates():
    df = pd.DataFrame({
        "Name": ["A", "A", "B"],
        "Symbol": ["A", "A", "B"],
        "Date": ["2020-01-01", "2020-01-01", "2020-01-01"],
        "Rating": ["AAA", "AAA", "BB"],
        "currentRatio": [1.0, 1.0, 2.0],
    })
    out, report = sb.clean_dataframe(df)
    assert len(out) == 2
    assert report["duplicate_rows_dropped"] == 1


def test_clean_dataframe_nulls_impossible_negatives():
    df = pd.DataFrame({
        "Name": ["A", "B"],
        "Symbol": ["A", "B"],
        "Date": ["2020-01-01", "2020-01-01"],
        "Rating": ["AAA", "BB"],
        "currentRatio": [-1.0, 2.0],  # currentRatio can't be negative
    })
    out, report = sb.clean_dataframe(df)
    assert report["impossible_negatives_nulled"] == 1
    assert pd.isna(out.loc[out["Name"] == "A", "currentRatio"]).all()


def _synthetic_dataset(n_companies=40, rows_per_company=3, seed=0):
    """A small synthetic dataset shaped like the real one: multiple rows
    (years) per company, a Sector column, and a handful of numeric ratios.
    """
    rng = np.random.default_rng(seed)
    grades = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
    sectors = ["Technology", "Energy", "Finance", "Health Care"]

    rows = []
    for company_idx in range(n_companies):
        symbol = f"CO{company_idx}"
        sector = sectors[company_idx % len(sectors)]
        for year in range(rows_per_company):
            rows.append({
                "Name": symbol,
                "Symbol": symbol,
                "Rating Agency Name": "TestAgency",
                "Date": f"{2015 + year}-01-01",
                "Rating": grades[(company_idx + year) % len(grades)],
                "Sector": sector,
                "currentRatio": rng.uniform(0.5, 3.0),
                "quickRatio": rng.uniform(0.3, 2.5),
                "cashRatio": rng.uniform(0.1, 1.5),
                "returnOnAssets": rng.uniform(-0.1, 0.2),
                "debtEquityRatio": rng.uniform(0.1, 5.0),
                "netProfitMargin": rng.uniform(-0.2, 0.3),
                "assetTurnover": rng.uniform(0.1, 2.0),
            })
    return pd.DataFrame(rows)


def test_make_split_keeps_companies_on_one_side():
    df = _synthetic_dataset()
    y = df["Rating"].apply(sb.group_rating)
    groups = sb.extract_groups(df)
    X_train, X_test, y_train, y_test, strategy = sb.make_split(df, y, groups=groups)

    assert strategy == "grouped_stratified"
    train_companies = set(X_train["Symbol"])
    test_companies = set(X_test["Symbol"])
    assert train_companies.isdisjoint(test_companies), (
        "A company's rows leaked across both sides of the split"
    )


def test_run_fair_baseline_end_to_end_shape():
    df = _synthetic_dataset()
    result = sb.run_fair_baseline(DecisionTreeClassifier(random_state=sb.RANDOM_STATE), df)

    assert result is not None
    metrics = result["metrics"]
    for key in (
        "accuracy", "precisionWeighted", "recallWeighted", "f1Weighted",
        "precisionMacro", "recallMacro", "f1Macro", "recallPerClass",
        "confusionMatrix", "splitStrategy", "testSamples", "trainSamples",
    ):
        assert key in metrics

    n_labels = len(metrics["confusionMatrix"]["labels"])
    cm_values = metrics["confusionMatrix"]["values"]
    assert len(cm_values) == n_labels
    assert all(len(row) == n_labels for row in cm_values)
    assert 0.0 <= metrics["accuracy"] <= 1.0
