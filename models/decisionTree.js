(function (root, factory) {
  const model = factory();
  const moduleExport = {
    name: "Decision Tree",
    route: "/predict/decision-tree",
    scriptPath: "python_models/decsion_tree_stuff/predict_decision_tree.py",
    data: model
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = moduleExport;
  }

  if (root) {
    root.MODEL_LIBRARY = root.MODEL_LIBRARY || {};
    root.MODEL_LIBRARY["Decision Tree"] = model;
  }
})(typeof window !== "undefined" ? window : globalThis, function () {
  return {
    tag: "Decision Tree",
    metrics: {
      accuracy: "0.5616",
      precision: "0.5691",
      recall: "0.5616",
      f1: "0.5599",
      strength: "Best at separating Investment-Low and Speculative classes in the tuned tree.",
      weakness: "Distressed is still the hardest class because the dataset is small and imbalanced."
    },
    matrix: [
      [42, 46, 11, 0],
      [21, 81, 30, 2],
      [8, 40, 103, 8],
      [1, 1, 10, 2]
    ],
    shap: [
      ["Current ratio", 92],
      ["Operating cash flow sales ratio", 84],
      ["Gross profit margin", 76],
      ["Operating cash flow per share", 68],
      ["Return on assets", 60],
      ["Quick ratio", 52]
    ],
    shapStory: {
      positive: ["current ratio", "operating cash flow sales ratio", "gross profit margin"],
      negative: ["return on assets", "quick ratio", "net profit margin"]
    }
  };
});
