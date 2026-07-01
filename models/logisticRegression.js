(function (root) {
  // Define browser-compatible and Node-compatible metadata for rendering in dashboard
  const modelData = {
    tag: "LogisticRegression",
    trained: false,
    metrics: null,
    labels: [],
    matrix: [],
    shap: [],
    shapStory: null
  };

  const moduleExport = {
    name: "LogisticRegression",
    route: "/predict/logistic-regression",
    scriptPath: "python_models/logistic_regression_stuff/predict_logistic.py",
    data: modelData
  };

  // Export for Node.js
  if (typeof module !== "undefined" && module.exports) {
    module.exports = moduleExport;
  }

  // Register globally for the Browser
  if (root) {
    root.MODEL_LIBRARY = root.MODEL_LIBRARY || {};
    root.MODEL_LIBRARY["LogisticRegression"] = modelData;
  }
})(typeof window !== "undefined" ? window : null);
