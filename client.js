const appState = {
  fileName: "",
  file: null,
  model: "Decision Tree",
  uploaded: false
};

const modelData = window.MODEL_LIBRARY || {};

// Canonical rating grouping — must stay in sync with group_rating() in
// preprocessing.py (XGBoost), predict_decision_tree.py, and
// predict_random_forest.py. CCC/CC are grouped with Distressed (not
// Speculative) so the Distressed tier has enough support to be measurable;
// see docs/TECHNICAL_REPORT.md §2.3 for the full justification.
const ratingRuleEngine = {
  "AAA": "Investment-High",
  "AA": "Investment-High",
  "A": "Investment-High",
  "BBB": "Investment-Low",
  "BB": "Speculative",
  "B": "Speculative",
  "CCC": "Distressed",
  "CC": "Distressed",
  "C": "Distressed",
  "D": "Distressed"
};

function mapRatingToGroup(rating) {
  return ratingRuleEngine[rating] || "Unknown";
}

function initApp() {
  const uploadScreen = document.getElementById("uploadScreen");
  const modelScreen = document.getElementById("modelScreen");
  const resultsScreen = document.getElementById("resultsScreen");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const continueBtn = document.getElementById("continueBtn");
  const backToUploadBtn = document.getElementById("backToUploadBtn");
  const backToModelBtn = document.getElementById("backToModelBtn");
  const dropzone = document.getElementById("dropzone");
  const selectedFileName = document.getElementById("selectedFileName");
  const selectedFileMeta = document.getElementById("selectedFileMeta");
  const modelFileName = document.getElementById("modelFileName");
  const modelBtnGroup = document.getElementById("modelBtnGroup");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const selectedModelTag = document.getElementById("selectedModelTag");
  const performanceStats = document.getElementById("performanceStats");
  const performanceStrength = document.getElementById("performanceStrength");
  const performanceWeakness = document.getElementById("performanceWeakness");
  const shapSummary = document.getElementById("shapSummary");
  const shapNarrative = document.getElementById("shapNarrative");
  const matrixGrid = document.getElementById("matrixGrid");
  const shapBars = document.getElementById("shapBars");
  const performanceLabel = document.getElementById("performanceLabel");
  const secondaryMetricsSection = document.getElementById("secondaryMetricsSection");
  const secondaryMetricsNote = document.getElementById("secondaryMetricsNote");
  const secondaryStats = document.getElementById("secondaryStats");
  const ratioInputs = document.querySelectorAll("[data-ratio-feature]");
  const ratioForm = document.getElementById("ratioForm");
  const predictionResult = document.getElementById("predictionResult");

  function formatPredictionLabel(label) {
    return String(label).replace(/_/g, "-");
  }

  function getBackendErrorMessage(result, fallback) {
    if (!result || typeof result !== "object") {
      return fallback;
    }

    const parts = [result.error, result.details, result.stderr].filter(Boolean);
    if (!parts.length) {
      return fallback;
    }

    return parts.join(" ");
  }

  function showScreen(screen) {
    [uploadScreen, modelScreen, resultsScreen].forEach((node) => node.classList.remove("active"));
    screen.classList.add("active");
  }

  function formatFileMeta(file) {
    const sizeKb = Math.max(1, Math.round(file.size / 1024));
    const typeLabel = file.type ? file.type : "Spreadsheet";
    return `${typeLabel} - ${sizeKb} KB`;
  }

  function setFileState(file) {
    appState.fileName = file.name;
    appState.file = file;
    appState.uploaded = true;
    selectedFileName.textContent = file.name;
    selectedFileMeta.textContent = formatFileMeta(file);
    modelFileName.textContent = file.name;
    continueBtn.disabled = false;
  }

  function renderMatrix(model) {
    matrixGrid.innerHTML = "";
    const values = modelData[model]?.matrix || [];
    const labels = modelData[model]?.labels || values.map((_, index) => `Class ${index + 1}`);

    if (!values.length) {
      matrixGrid.style.gridTemplateColumns = "1fr";
      matrixGrid.innerHTML = `
        <div class="cell">
          <strong>-</strong>
          <span>Train the model to generate a confusion matrix.</span>
        </div>
      `;
      return;
    }

    const columnCount = labels.length || values.length || 1;
    matrixGrid.style.gridTemplateColumns = `repeat(${columnCount}, minmax(0, 1fr))`;

    values.flat().forEach((value, index) => {
      const row = Math.floor(index / columnCount);
      const col = index % columnCount;
      const cell = document.createElement("div");
      cell.className = "cell";
      if (row === col) cell.classList.add("diag");
      cell.innerHTML = `
        <strong>${value}</strong>
        <span>${labels[row]}<br>predicted as ${labels[col]}</span>
      `;
      matrixGrid.appendChild(cell);
    });
  }

  function renderShap(model) {
    shapBars.innerHTML = "";

    function getEffectDirection(effect) {
      if (typeof effect === "number") {
        return effect < 0 ? -1 : 1;
      }

      if (typeof effect === "string") {
        return effect.toLowerCase().includes("away") ? -1 : 1;
      }

      return 1;
    }

    const shapValues = (modelData[model]?.featureImportance || modelData[model]?.shap || [])
      .map((item) => {
        if (Array.isArray(item)) {
          return {
            feature: item[0],
            value: Number(item[1]) || 0,
            effect: item[2] || null
          };
        }

        return {
          feature: item.feature,
          value: Number(item.importance ?? item.value) || 0,
          effect: item.effect || null
        };
      })
      .filter((item) => item.feature);

    if (!shapValues.length) {
      shapBars.innerHTML = `
        <div class="bar-row">
          <label>Train the model to generate feature importance.</label>
          <div class="bar-track"><div class="bar-fill" style="width: 0%;"></div></div>
          <strong>-</strong>
        </div>
      `;
      shapNarrative.innerHTML = `
        <h5>Why this class was predicted</h5>
        <p>Feature importance appears after the model is trained on the uploaded dataset.</p>
      `;
      return;
    }

    const rankedValues = shapValues
      .map((item) => ({
        ...item,
        absValue: Math.abs(item.value)
      }))
      .sort((left, right) => right.absValue - left.absValue)
      .slice(0, 8);

    const maxValue = rankedValues[0]?.absValue || 0;
    const totalValue = rankedValues.reduce((sum, item) => sum + item.absValue, 0) || 1;
    const topItem = rankedValues[0];
    const secondItem = rankedValues[1];

    if (topItem) {
      const topShare = Math.round((topItem.absValue / totalValue) * 100);
      const runnerUpShare = secondItem ? Math.round((secondItem.absValue / topItem.absValue) * 100) : null;
      const closenessNote = runnerUpShare && runnerUpShare >= 85
        ? ` The next strongest feature is very close at ${runnerUpShare}% of the top driver.`
        : "";
      shapSummary.textContent = `Top SHAP driver for ${model}: ${topItem.feature} (${topShare}% of total displayed impact).${closenessNote}`;
    }

    rankedValues.forEach((item, index) => {
      const label = item.feature;
      const value = item.absValue;
      const direction = getEffectDirection(item.effect);
      const relativeWidth = maxValue ? (value / maxValue) * 100 : 0;
      const relativeStrength = maxValue ? Math.round((value / maxValue) * 100) : 0;

      const row = document.createElement("div");
      row.className = "bar-row";

      const barColorStyle = direction === -1
        ? "background: linear-gradient(90deg, var(--accent-2), #ffa47a);"
        : "background: linear-gradient(90deg, var(--accent), var(--accent-3));";

      const directionLabel = direction === -1 ? "Pulls Away" : "Pushes Toward";
      const badgeStyle = direction === -1
        ? "color: var(--accent-2); font-weight: 600;"
        : "color: var(--accent); font-weight: 600;";
      const isTop = index === 0;
      const emphasisStyle = isTop
        ? "font-weight: 700; color: var(--text);"
        : "font-weight: 500;";
      
      row.innerHTML = `
        <label style="display: flex; flex-direction: column; gap: 2px;">
          <span style="${emphasisStyle}">${label}${isTop ? ' <span style="margin-left: 6px; padding: 2px 8px; border-radius: 999px; background: rgba(0,184,169,0.14); color: var(--accent); font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;">Top driver</span>' : ''}</span>
          <span style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; ${badgeStyle}">${directionLabel}</span>
        </label>
        <div class="bar-track"><div class="bar-fill" style="width: ${relativeWidth}%; opacity: ${isTop ? 1 : Math.max(0.55, 0.92 - index * 0.05)}; ${barColorStyle}"></div></div>
        <strong>${value.toFixed(4)}%<br><span style="font-size: 0.68rem; font-weight: 600; color: var(--accent);">${isTop ? "100% of top" : `${relativeStrength}% of top`}</span></strong>
      `;
      shapBars.appendChild(row);
    });

    const story = modelData[model]?.shapStory;
    if (story) {
      shapNarrative.innerHTML = `
        <h5>Why this class was predicted</h5>
        <p>
          The model leaned toward the predicted risk class because ${story.positive.join(", ")} pushed the score up,
          while ${story.negative.join(", ")} pulled it back.
        </p>
      `;
    } else {
      shapNarrative.innerHTML = `
        <h5>Why this class was predicted</h5>
        <p>
          The SHAP plot now ranks the strongest financial drivers first, so the most influential column is easier to spot.
        </p>
      `;
    }
  }

  function renderMetrics(model) {
    const entry = modelData[model] || {};
    const metrics = entry.metrics;
    const cvMetrics = entry.cvMetrics;
    const singleSplitMetrics = entry.singleSplitMetrics;

    if (!metrics) {
      performanceLabel.textContent = "Key evaluation metrics for the selected model, plus a short read on the main strength and weakness.";
      performanceStats.innerHTML = `
        <div class="mini-card">
          <h5>Accuracy</h5>
          <p>-</p>
        </div>
        <div class="mini-card">
          <h5>Precision</h5>
          <p>-</p>
        </div>
        <div class="mini-card">
          <h5>Recall</h5>
          <p>-</p>
        </div>
        <div class="mini-card">
          <h5>F1-score</h5>
          <p>-</p>
        </div>
      `;
      performanceStrength.textContent = "Train the model to calculate metrics from the uploaded dataset.";
      performanceWeakness.textContent = "No static Random Forest metrics are shown before training.";
      shapSummary.textContent = `Feature importance for ${model} appears after training.`;
      secondaryMetricsSection.style.display = "none";
      return;
    }

    // Primary card: for models that report cross-validated metrics (currently
    // XGBoost), this is the CV mean +/- std — the authoritative, low-variance
    // reported accuracy. For models without a CV cache yet, this falls back
    // to whatever the backend put in `metrics` (a single-split figure, with
    // its own explicit caveat baked into the strength/weakness text).
    if (cvMetrics) {
      performanceLabel.textContent =
        `Reported accuracy (primary): ${cvMetrics.label}. This is the authoritative, ` +
        `cross-validated figure — not a single train/test split.`;
      performanceStats.innerHTML = `
        <div class="mini-card">
          <h5>Accuracy (CV mean)</h5>
          <p>${metrics.accuracy} ± ${cvMetrics.accuracyStd}</p>
        </div>
        <div class="mini-card">
          <h5>Macro F1 (CV mean)</h5>
          <p>${metrics.f1} ± ${cvMetrics.f1Std}</p>
        </div>
        <div class="mini-card">
          <h5>CV folds</h5>
          <p>${cvMetrics.cvFolds}</p>
        </div>
        <div class="mini-card">
          <h5>Split strategy</h5>
          <p>Grouped, company-level</p>
        </div>
      `;
    } else {
      performanceLabel.textContent = "Key evaluation metrics for the selected model, plus a short read on the main strength and weakness.";
      performanceStats.innerHTML = `
        <div class="mini-card">
          <h5>Accuracy</h5>
          <p>${metrics.accuracy}</p>
        </div>
        <div class="mini-card">
          <h5>Precision</h5>
          <p>${metrics.precision ?? "-"}</p>
        </div>
        <div class="mini-card">
          <h5>Recall</h5>
          <p>${metrics.recall ?? "-"}</p>
        </div>
        <div class="mini-card">
          <h5>F1-score</h5>
          <p>${metrics.f1}</p>
        </div>
      `;
    }

    performanceStrength.textContent = metrics.strength;
    performanceWeakness.textContent = metrics.weakness;
    shapSummary.textContent = `SHAP explains which features pushed the company toward its predicted risk class for ${model}.`;

    // Secondary card: single-split demo metrics, only shown (collapsed by
    // default via <details>) when the backend explicitly separated it out.
    // Labeled clearly so it never reads as competing with the primary figure.
    if (singleSplitMetrics) {
      secondaryMetricsSection.style.display = "block";
      secondaryMetricsNote.textContent = singleSplitMetrics.note || singleSplitMetrics.label || "";
      secondaryStats.innerHTML = `
        <div class="mini-card">
          <h5>Accuracy</h5>
          <p>${singleSplitMetrics.accuracy}</p>
        </div>
        <div class="mini-card">
          <h5>Precision</h5>
          <p>${singleSplitMetrics.precision ?? "-"}</p>
        </div>
        <div class="mini-card">
          <h5>Recall</h5>
          <p>${singleSplitMetrics.recall ?? "-"}</p>
        </div>
        <div class="mini-card">
          <h5>F1-score</h5>
          <p>${singleSplitMetrics.f1}</p>
        </div>
      `;
    } else {
      secondaryMetricsSection.style.display = "none";
    }
  }

  // Returns the currently active model name from the button group.
  function getSelectedModel() {
    const active = modelBtnGroup.querySelector(".model-btn.active");
    return active ? active.dataset.model : "Decision Tree";
  }

  // Activates the clicked model button and deactivates the rest.
  function setActiveModelBtn(btn) {
    modelBtnGroup.querySelectorAll(".model-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
  }

  function runAnalysis() {
    appState.model = getSelectedModel();
    selectedModelTag.textContent = `Model: ${appState.model}`;
    renderMetrics(appState.model);
    renderMatrix(appState.model);
    renderShap(appState.model);
    showScreen(resultsScreen);
  }

  function updateActionLabel() {
    const model = getSelectedModel();
    if (model === "Decision Tree" || model === "Random Forest" || model === "XGBoost" || model === "Logistic Regression") {
      analyzeBtn.textContent = "Train / Predict";
      ratioForm.style.display = "none";
    } else {
      analyzeBtn.textContent = "Run analysis";
      ratioForm.style.display = "none";
    }
  }

  function getRatioPayload() {
    const payload = {};

    for (const input of ratioInputs) {
      const fieldName = input.dataset.ratioFeature;
      if (input.tagName === "SELECT") {
        if (!input.value) {
          throw new Error("Sector is required for ratio-based predictions.");
        }
        payload[fieldName] = input.value;
        continue;
      }

      const value = Number(input.value);
      if (!Number.isFinite(value)) {
        throw new Error(`${input.labels[0]?.textContent || input.name} must be a number.`);
      }
      payload[fieldName] = value;
    }

    return payload;
  }

  function arrayBufferToBase64(buffer) {
    let binary = "";
    const bytes = new Uint8Array(buffer);
    const chunkSize = 0x8000;

    for (let i = 0; i < bytes.length; i += chunkSize) {
      const chunk = bytes.subarray(i, i + chunkSize);
      binary += String.fromCharCode(...chunk);
    }

    return btoa(binary);
  }

  function readFileAsText(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ""));
      reader.onerror = () => reject(reader.error || new Error("Unable to read file."));
      reader.readAsText(file);
    });
  }

  function readFileAsArrayBuffer(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error || new Error("Unable to read file."));
      reader.readAsArrayBuffer(file);
    });
  }

  async function buildDecisionTreePayload() {
    if (!appState.file) {
      throw new Error("Please upload a CSV or Excel file before training a model.");
    }

    const fileName = appState.file.name || "dataset.csv";
    const extension = fileName.split(".").pop().toLowerCase();

    if (extension === "csv") {
      const text = await readFileAsText(appState.file);
      return {
        fileName,
        fileEncoding: "utf8",
        fileData: text
      };
    }

    const buffer = await readFileAsArrayBuffer(appState.file);
    return {
      fileName,
      fileEncoding: "base64",
      fileData: arrayBufferToBase64(buffer)
    };
  }

  async function predictDecisionTree() {
    let payload;

    try {
      payload = await buildDecisionTreePayload();
    } catch (error) {
      predictionResult.textContent = error.message;
      predictionResult.className = "prediction-result error";
      return;
    }

    analyzeBtn.disabled = true;
    predictionResult.textContent = "Training Decision Tree on uploaded data...";
    predictionResult.className = "prediction-result";

    try {
      const response = await fetch("/predict/decision-tree", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Decision Tree training failed.");
      }

      if (result.modelData) {
        modelData["Decision Tree"] = result.modelData;
      }

      predictionResult.textContent = `Prediction: ${formatPredictionLabel(result.samplePrediction?.prediction || result.prediction || "Unknown")}`;
      predictionResult.className = "prediction-result success";
      appState.model = "Decision Tree";
      selectedModelTag.textContent = `Model: Decision Tree · Prediction: ${formatPredictionLabel(result.samplePrediction?.prediction || result.prediction || "Unknown")}`;
      renderMetrics("Decision Tree");
      renderMatrix("Decision Tree");
      renderShap("Decision Tree");
      showScreen(resultsScreen);
    } catch (error) {
      predictionResult.textContent = error.message || "Unable to train the Decision Tree.";
      predictionResult.className = "prediction-result error";
    } finally {
      analyzeBtn.disabled = false;
    }
  }

  async function predictRandomForest() {
    let payload;

    try {
      payload = await buildDecisionTreePayload();
    } catch (error) {
      predictionResult.textContent = error.message;
      predictionResult.className = "prediction-result error";
      return;
    }

    analyzeBtn.disabled = true;
    predictionResult.textContent = "Training Random Forest on uploaded data...";
    predictionResult.className = "prediction-result";

    try {
      const response = await fetch("/predict/random-forest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(getBackendErrorMessage(result, "Random Forest prediction failed."));
      }

      if (result.modelData) {
        modelData["Random Forest"] = result.modelData;
      }

      predictionResult.textContent = "Model trained and dataset evaluated successfully.";
      predictionResult.className = "prediction-result success";
      appState.model = "Random Forest";
      renderMetrics("Random Forest");
      renderMatrix("Random Forest");
      renderShap("Random Forest");
      showScreen(resultsScreen);
      selectedModelTag.textContent = `Model: Random Forest · Prediction: ${formatPredictionLabel(result.prediction)}`;
    } catch (error) {
      predictionResult.textContent = error.message || "Unable to get a prediction.";
      predictionResult.className = "prediction-result error";
    } finally {
      analyzeBtn.disabled = false;
    }
  }

  async function predictLogisticRegression() {
    let payload;

    try {
      payload = await buildDecisionTreePayload();
    } catch (error) {
      predictionResult.textContent = error.message;
      predictionResult.className = "prediction-result error";
      return;
    }

    analyzeBtn.disabled = true;
    predictionResult.textContent = "Training Logistic Regression on uploaded data...";
    predictionResult.className = "prediction-result";

    try {
      const response = await fetch("/predict/logistic-regression", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(getBackendErrorMessage(result, "Logistic Regression prediction failed."));
      }

      if (result.modelData) {
        modelData["Logistic Regression"] = result.modelData;
      }

      predictionResult.textContent = "Model trained and dataset evaluated successfully.";
      predictionResult.className = "prediction-result success";
      appState.model = "Logistic Regression";
      selectedModelTag.textContent = `Model: Logistic Regression · Prediction: ${formatPredictionLabel(result.samplePrediction?.prediction || result.prediction || "Unknown")}`;
      renderMetrics("Logistic Regression");
      renderMatrix("Logistic Regression");
      renderShap("Logistic Regression");
      showScreen(resultsScreen);
    } catch (error) {
      predictionResult.textContent = error.message || "Unable to get a prediction.";
      predictionResult.className = "prediction-result error";
    } finally {
      analyzeBtn.disabled = false;
    }
  }

  async function predictXgboost() {
    let payload;

    try {
      payload = await buildDecisionTreePayload();
    } catch (error) {
      predictionResult.textContent = error.message;
      predictionResult.className = "prediction-result error";
      return;
    }

    analyzeBtn.disabled = true;
    predictionResult.textContent = "Training XGBoost on uploaded data…";
    predictionResult.className = "prediction-result";

    try {
      const response = await fetch("/predict/xgboost", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(getBackendErrorMessage(result, "XGBoost prediction failed."));
      }

      if (result.modelData) {
        modelData["XGBoost"] = result.modelData;
      }

      predictionResult.textContent = "Model trained and dataset evaluated successfully.";
      predictionResult.className = "prediction-result success";
      appState.model = "XGBoost";
      selectedModelTag.textContent = `Model: XGBoost · Prediction: ${formatPredictionLabel(result.prediction || "Unknown")}`;
      renderMetrics("XGBoost");
      renderMatrix("XGBoost");
      renderShap("XGBoost");
      showScreen(resultsScreen);
    } catch (error) {
      predictionResult.textContent = error.message || "Unable to get a prediction.";
      predictionResult.className = "prediction-result error";
    } finally {
      analyzeBtn.disabled = false;
    }
  }

  browseBtn.addEventListener("click", (event) => {
    event.stopPropagation();
  });

  browseBtn.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    setFileState(file);
  });

  continueBtn.addEventListener("click", () => {
    if (!appState.uploaded) return;
    showScreen(modelScreen);
  });

  backToUploadBtn.addEventListener("click", () => {
    showScreen(uploadScreen);
  });

  backToModelBtn.addEventListener("click", () => {
    showScreen(modelScreen);
  });

  analyzeBtn.addEventListener("click", () => {
    const model = getSelectedModel();
    if (model === "Decision Tree") {
      predictDecisionTree();
      return;
    }
    if (model === "Random Forest") {
      predictRandomForest();
      return;
    }
    if (model === "XGBoost") {
      predictXgboost();
      return;
    }
    if (model === "Logistic Regression") {
      predictLogisticRegression();
      return;
    }
    if (!appState.uploaded) return;
    runAnalysis();
  });

  // Update button label whenever a model button is clicked.
  modelBtnGroup.addEventListener("click", (event) => {
    const btn = event.target.closest(".model-btn");
    if (!btn) return;
    setActiveModelBtn(btn);
    updateActionLabel();
  });

  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (!file) return;
    setFileState(file);
  });

  dropzone.addEventListener("click", (event) => {
    if (event.target.closest("button, label")) return;
    fileInput.click();
  });

  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  renderMatrix("Decision Tree");
  renderShap("Decision Tree");
  renderMetrics("Decision Tree");
  updateActionLabel();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}