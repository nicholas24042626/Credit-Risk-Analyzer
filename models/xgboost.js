(function (root) {
  // Define browser-compatible and Node-compatible metadata for rendering in dashboard
  const modelData = {
    tag: "XGBoost",
    trained: false,
    metrics: null,
    labels: [],
    matrix: [],
    shap: [],
    shapStory: null
  };

  const moduleExport = {
    name: "XGBoost",
    route: "/predict/xgboost",
    scriptPath: "python_models/xgboost_stuff/predict_xgboost.py",
    data: modelData
  };

  // Export for Node.js
  if (typeof module !== "undefined" && module.exports) {
    module.exports = moduleExport;
  }

  // Register globally for the Browser
  if (root) {
    root.MODEL_LIBRARY = root.MODEL_LIBRARY || {};
    root.MODEL_LIBRARY["XGBoost"] = modelData;
  }
})(typeof window !== "undefined" ? window : null);
