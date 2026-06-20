const appState = {
  fileName: "",
  model: "Decision Tree",
  uploaded: false
};

const modelData = {
  "Decision Tree": {
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
  },
  "Random Forest": {
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
  },
  "Logistic Regression": {
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
  },
  "XGBoost": {
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
  }
};

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
    if (!appState.uploaded) return;
    runAnalysis();
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
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initApp);
} else {
  initApp();
}
