const appState = {
  fileName: "",
  file: null,
  model: "Decision Tree",
  uploaded: false
};

const modelData = window.MODEL_LIBRARY || {};

const ratingRuleEngine = {
  "AAA": "Investment-High",
  "AA": "Investment-High",
  "A": "Investment-High",
  "BBB": "Investment-Low",
  "BB": "Speculative",
  "B": "Speculative",
  "CCC": "Speculative",
  "CC": "Speculative",
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
  const modelSelect = document.getElementById("modelSelect");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const selectedModelTag = document.getElementById("selectedModelTag");
  const performanceStats = document.getElementById("performanceStats");
  const performanceStrength = document.getElementById("performanceStrength");
  const performanceWeakness = document.getElementById("performanceWeakness");
  const shapSummary = document.getElementById("shapSummary");
  const shapNarrative = document.getElementById("shapNarrative");
  const matrixGrid = document.getElementById("matrixGrid");
  const shapBars = document.getElementById("shapBars");
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
    const shapValues = modelData[model]?.shap || [];

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

    shapValues.forEach((item, index) => {
      const label = item[0];
      const value = item[1];
      const direction = item[2] !== undefined ? item[2] : 1;
      
      const row = document.createElement("div");
      row.className = "bar-row";
      
      const barColorStyle = direction === -1 
        ? 'background: linear-gradient(90deg, var(--accent-2), #ffa47a);' 
        : 'background: linear-gradient(90deg, var(--accent), var(--accent-3));';
        
      const directionLabel = direction === -1 ? 'Pulls Away' : 'Pushes Toward';
      const badgeStyle = direction === -1 ? 'color: var(--accent-2); font-weight: 500;' : 'color: var(--accent); font-weight: 500;';
      
      row.innerHTML = `
        <label style="display: flex; flex-direction: column; gap: 2px;">
          <span>${label}</span>
          <span style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em; ${badgeStyle}">${directionLabel}</span>
        </label>
        <div class="bar-track"><div class="bar-fill" style="width: ${value}%; opacity: ${0.78 + index * 0.02}; ${barColorStyle}"></div></div>
        <strong>${value}%</strong>
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
          The SHAP plot shows which financial ratios pushed the company toward the selected risk class and which ones pulled away.
        </p>
      `;
    }
  }

  function renderMetrics(model) {
    const metrics = modelData[model]?.metrics;

    if (!metrics) {
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
      return;
    }

    performanceStats.innerHTML = `
      <div class="mini-card">
        <h5>Accuracy</h5>
        <p>${metrics.accuracy}</p>
      </div>
      <div class="mini-card">
        <h5>Precision</h5>
        <p>${metrics.precision}</p>
      </div>
      <div class="mini-card">
        <h5>Recall</h5>
        <p>${metrics.recall}</p>
      </div>
      <div class="mini-card">
        <h5>F1-score</h5>
        <p>${metrics.f1}</p>
      </div>
    `;
    performanceStrength.textContent = metrics.strength;
    performanceWeakness.textContent = metrics.weakness;
    shapSummary.textContent = `SHAP explains which features pushed the company toward its predicted risk class for ${model}.`;
  }

  function runAnalysis() {
    appState.model = modelSelect.value;
    selectedModelTag.textContent = `Model: ${appState.model}`;
    renderMetrics(appState.model);
    renderMatrix(appState.model);
    renderShap(appState.model);
    showScreen(resultsScreen);
  }

  function updateActionLabel() {
    if (modelSelect.value === "Decision Tree" || modelSelect.value === "Random Forest" || modelSelect.value === "XGBoost") {
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
    predictionResult.textContent = "Running XGBoost prediction on uploaded data…";
    predictionResult.className = "prediction-result";

    try {
      const response = await fetch("/predict/xgboost", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "XGBoost prediction failed.");
      }

      if (result.modelData) {
        modelData["XGBoost"] = result.modelData;
      }

      predictionResult.textContent = `Prediction: ${formatPredictionLabel(result.prediction || "Unknown")}`;
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
    if (modelSelect.value === "Decision Tree") {
      predictDecisionTree();
      return;
    }
    if (modelSelect.value === "Random Forest") {
      predictRandomForest();
      return;
    }
    if (modelSelect.value === "XGBoost") {
      predictXgboost();
      return;
    }
    if (!appState.uploaded) return;
    runAnalysis();
  });

  modelSelect.addEventListener("change", updateActionLabel);

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
