(function (root) {
  const model = {
    name: "Random Forest",
    route: "/predict/random-forest",
    description: "Runs the Python-trained Random Forest credit risk predictor.",
    data: {
      tag: "Random Forest",
      metrics: {
        accuracy: "0.66",
        precision: "0.66",
        recall: "0.66",
        f1: "0.65",
        strength: "Captures non-linear relationships across the financial ratios.",
        weakness: "Performance can vary across the smaller credit-risk classes."
      },
      matrix: [
        [89, 43, 16, 0],
        [39, 114, 48, 0],
        [10, 29, 198, 1],
        [1, 0, 19, 2]
      ],
      shap: [
        ["Current ratio", 89],
        ["Debt-to-equity ratio", 81],
        ["Net profit margin", 74],
        ["Return on assets", 67],
        ["Operating cash flow / sales", 61],
        ["Enterprise value multiple", 52]
      ]
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
