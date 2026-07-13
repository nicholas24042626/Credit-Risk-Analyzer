import sys
from pathlib import Path

PYTHON_MODELS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PYTHON_MODELS_DIR))
sys.path.insert(0, str(PYTHON_MODELS_DIR / "xgboost_stuff"))
sys.path.insert(0, str(PYTHON_MODELS_DIR / "decision_tree_stuff"))
sys.path.insert(0, str(PYTHON_MODELS_DIR / "random_forest_stuff"))
sys.path.insert(0, str(PYTHON_MODELS_DIR / "logistic_regression_stuff"))
