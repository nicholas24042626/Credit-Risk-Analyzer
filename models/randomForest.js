window.MODEL_LIBRARY = window.MODEL_LIBRARY || {};

window.MODEL_LIBRARY["Random Forest"] = {
  tag: "Random Forest",
  metrics: {
    accuracy: "0.83",
    precision: "0.82",
    recall: "0.80",
    f1: "0.81",
    strength: "Strong overall balance and stable predictions across classes.",
    weakness: "Less transparent than a single tree when explaining an individual company."
  },
  matrix: [
    [60, 2, 1, 1],
    [4, 52, 3, 2],
    [1, 3, 43, 2],
    [0, 2, 4, 33]
  ],
  shap: [
    ["Debt ratio", 88],
    ["Enterprise value multiple", 79],
    ["Current ratio", 74],
    ["Operating profit margin", 66],
    ["Return on equity", 59],
    ["Sector", 45]
  ]
};
