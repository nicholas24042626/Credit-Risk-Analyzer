(function (root, factory) {
  const model = factory();
  const moduleExport = model;

  if (typeof module !== "undefined" && module.exports) {
    module.exports = moduleExport;
  }

  if (root) {
    root.MODEL_LIBRARY = root.MODEL_LIBRARY || {};
    root.MODEL_LIBRARY["Decision Tree"] = model;
  }
})(typeof window !== "undefined" ? window : globalThis, function () {
  return {
    name: "Decision Tree",
    route: "/predict/decision-tree",
    scriptPath: "python_models/decision_tree_stuff/predict_decision_tree.py"
  };
});
