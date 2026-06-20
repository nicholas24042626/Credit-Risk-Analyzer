window.MODEL_LIBRARY = window.MODEL_LIBRARY || {};

window.MODEL_LIBRARY["XGBoost"] = {
  tag: "XGBoost",
  metrics: {
    accuracy: "0.86",
    precision: "0.85",
    recall: "0.84",
    f1: "0.84",
    strength: "Captures complex interactions and usually performs strongest overall.",
    weakness: "Can be harder to explain if the audience wants a simple rule trace."
  },
  matrix: [
    [61, 2, 1, 0],
    [3, 54, 2, 2],
    [1, 2, 44, 2],
    [0, 1, 3, 35]
  ],
  shap: [
    ["Operating cash flow", 90],
    ["Debt ratio", 82],
    ["Current ratio", 79],
    ["Return on capital employed", 68],
    ["Gross profit margin", 59],
    ["Sector", 46]
  ]
};
