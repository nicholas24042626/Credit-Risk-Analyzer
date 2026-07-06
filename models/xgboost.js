(function (root) {
  const model = {
    name: "XGBoost",
    route: "/predict/xgboost",
    scriptPath: "python_models/xgboost_stuff/predict_xgboost.py",
    description: "Trains and evaluates a calibrated XGBoost credit risk predictor from the uploaded dataset.",
    data: {
      tag: "XGBoost",
      trained: false,
      metrics: null,
      labels: [],
      matrix: [],
      shap: [],
      shapStory: null
    }
  };

  // Export route settings to Node and dashboard data to the browser.
  if (typeof module !== "undefined" && module.exports) {
    module.exports = model;
  }

  if (root) {
    root.MODEL_LIBRARY = root.MODEL_LIBRARY || {};
    root.MODEL_LIBRARY[model.name] = model.data;
  }
})(typeof window !== "undefined" ? window : null);
