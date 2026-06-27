(function (root) {
  const model = {
    name: "Random Forest",
    route: "/predict/random-forest",
    scriptPath: "python_models/random_forest_stuff/predict_random_forest.py",
    description: "Trains and evaluates a Random Forest credit risk predictor from the uploaded dataset.",
    data: {
      tag: "Random Forest",
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
