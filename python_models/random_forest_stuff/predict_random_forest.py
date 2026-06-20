import json
import sys
from pathlib import Path

import joblib
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "random_forest_model.pkl"
FEATURES_PATH = SCRIPT_DIR / "random_forest_features.pkl"


def fail(message, details=None, status=1):
    payload = {"error": message}
    if details is not None:
        payload["details"] = details
    print(json.dumps(payload), file=sys.stderr)
    raise SystemExit(status)


def load_artifact(path):
    if not path.exists():
        fail("Required artifact is missing.", {"path": str(path)})

    try:
        return joblib.load(path)
    except Exception as exc:
        fail("Failed to load model artifact.", {
            "path": str(path),
            "details": str(exc)
        })


def parse_request_payload():
    # Support both direct command-line tests and JSON sent by the Node server.
    raw_payload = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read().strip()

    if not raw_payload:
        fail("Missing JSON request argument.")

    try:
        return json.loads(raw_payload)
    except Exception as exc:
        fail("Invalid JSON request argument.", {"details": str(exc)})


def normalize_records(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]

    return [payload]


def build_matrix(records, feature_names):
    if not records:
        fail("No input records were provided.")

    rows = []

    for record in records:
        if not isinstance(record, dict):
            fail("Each input record must be a JSON object.")

        row = {}

        for feature in feature_names:
            value = record.get(feature)

            if value is None or value == "":
                value = 0

            try:
                row[feature] = float(value)
            except Exception:
                fail("Invalid feature value.", {
                    "feature": feature,
                    "value": value
                })

        rows.append(row)

    # Use DataFrame so feature names are kept
    return pd.DataFrame(rows, columns=feature_names)


def main():
    payload = parse_request_payload()

    model = load_artifact(MODEL_PATH)
    feature_names = load_artifact(FEATURES_PATH)

    if not isinstance(feature_names, (list, tuple)):
        fail("Feature artifact must contain a list of feature names.", {
            "path": str(FEATURES_PATH)
        })

    records = normalize_records(payload)
    matrix = build_matrix(records, feature_names)

    try:
        predictions = model.predict(matrix)
    except Exception as exc:
        fail("Prediction failed.", {"details": str(exc)})

    prediction_list = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)

    response = {
        "model": "Random Forest",
        "featureCount": len(feature_names),
        "prediction": prediction_list[0],
        "predictions": prediction_list
    }

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(matrix)
            response["probabilities"] = probabilities.tolist() if hasattr(probabilities, "tolist") else list(probabilities)
        except Exception:
            pass

    print(json.dumps(response))


if __name__ == "__main__":
    main()
