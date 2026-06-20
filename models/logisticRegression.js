window.MODEL_LIBRARY = window.MODEL_LIBRARY || {};

window.MODEL_LIBRARY["Logistic Regression"] = {
  tag: "Logistic Regression",
  metrics: {
    accuracy: "0.76",
    precision: "0.74",
    recall: "0.72",
    f1: "0.73",
    strength: "Simple, fast, and easy to explain with directional feature effects.",
    weakness: "May miss non-linear patterns in company risk signals."
  },
  matrix: [
    [54, 6, 2, 2],
    [5, 50, 4, 2],
    [3, 4, 39, 3],
    [1, 3, 5, 30]
  ],
  shap: [
    ["Current ratio", 86],
    ["Net profit margin", 73],
    ["Debt ratio", 69],
    ["Return on assets", 61],
    ["Cash per share", 56],
    ["Sector", 40]
  ]
};
