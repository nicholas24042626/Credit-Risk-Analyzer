(function (root) {
  const model = {
    name: "Logistic Regression",
    route: "/predict/logistic-regression",
    scriptPath: "python_models/logistic_regression_stuff/predict_logistic_regression.py",
    description: "Trains and evaluates a Logistic Regression credit risk predictor from the uploaded dataset.",
    data: {
      tag: "Logistic Regression",
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