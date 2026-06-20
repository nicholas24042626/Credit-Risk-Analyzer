const appState = {
  fileName: "",
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
  const predictionResult = document.getElementById("predictionResult");

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
    appState.uploaded = true;
    selectedFileName.textContent = file.name;
    selectedFileMeta.textContent = formatFileMeta(file);
    modelFileName.textContent = file.name;
    continueBtn.disabled = false;
  }

  function renderMatrix(model) {
    matrixGrid.innerHTML = "";
    const labels = ["Investment-High", "Investment-Low", "Speculative", "Distressed"];
    const values = modelData[model].matrix;

    values.flat().forEach((value, index) => {
      const row = Math.floor(index / 4);
      const col = index % 4;
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
    modelData[model].shap.forEach(([label, value], index) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <label>${label}</label>
        <div class="bar-track"><div class="bar-fill" style="width: ${value}%; opacity: ${0.78 + index * 0.03};"></div></div>
        <strong>${value}%</strong>
      `;
      shapBars.appendChild(row);
    });

    const story = modelData[model].shapStory;
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
    const metrics = modelData[model].metrics;
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
    analyzeBtn.textContent = modelSelect.value === "Random Forest" ? "Predict" : "Run analysis";
  }

  function getRandomForestPayload() {
    const payload = {};

    for (const input of ratioInputs) {
      const value = Number(input.value);
      if (!Number.isFinite(value)) {
        throw new Error(`${input.labels[0]?.textContent || input.name} must be a number.`);
      }
      payload[input.dataset.ratioFeature] = value;
    }

    return payload;
  }

  async function predictRandomForest() {
    let payload;

    try {
      payload = getRandomForestPayload();
    } catch (error) {
      predictionResult.textContent = error.message;
      predictionResult.className = "prediction-result error";
      return;
    }

    analyzeBtn.disabled = true;
    predictionResult.textContent = "Running Random Forest prediction…";
    predictionResult.className = "prediction-result";

    try {
      const response = await fetch("/predict/random-forest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Random Forest prediction failed.");
      }

      predictionResult.textContent = `Prediction: ${result.prediction}`;
      predictionResult.className = "prediction-result success";
      runAnalysis();
      selectedModelTag.textContent = `Model: Random Forest · Prediction: ${result.prediction}`;
    } catch (error) {
      predictionResult.textContent = error.message || "Unable to get a prediction.";
      predictionResult.className = "prediction-result error";
    } finally {
      analyzeBtn.disabled = false;
    }
  }

  browseBtn.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    setFileState(file);
  });

  continueBtn.addEventListener("click", () => {
    if (!appState.uploaded) return;
    showScreen(modelScreen);
  });

  analyzeBtn.addEventListener("click", () => {
    if (modelSelect.value === "Random Forest") {
      predictRandomForest();
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
    if (event.target.closest("button")) return;
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
