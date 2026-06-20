(function (root) {
  let predictXGBoost;

  // Define Node.js specific prediction logic
  if (typeof module !== "undefined" && module.exports) {
    const { spawn } = require("node:child_process");
    const path = require("node:path");
    const fs = require("node:fs");

    predictXGBoost = function(inputData) {
      return new Promise((resolve, reject) => {
        // Path to the companion Python script
        const scriptPath = path.join(__dirname, "..", "python_models", "xgboost_stuff", "predict_xgboost.py");
        
        // Use the virtual environment Python interpreter if available
        const venvPath = path.join(__dirname, "..", ".venv", "Scripts", "python.exe");
        const pythonExecutable = fs.existsSync(venvPath) ? venvPath : "python";

        // Spawn Python process, similar to the existing app.js structure
        const pythonProcess = spawn(pythonExecutable, [scriptPath, JSON.stringify(inputData)], { windowsHide: true });
        
        let stdoutData = "";
        let stderrData = "";

        pythonProcess.stdout.on("data", (data) => { 
          stdoutData += data.toString(); 
        });
        
        pythonProcess.stderr.on("data", (data) => { 
          stderrData += data.toString(); 
        });

        pythonProcess.on("error", (err) => {
          reject(new Error(`Failed to start Python process: ${err.message}`));
        });

        pythonProcess.on("close", (code) => {
          if (code !== 0) {
            reject(new Error(`Python process failed with exit code ${code}. Stderr: ${stderrData.trim()}`));
            return;
          }
          try {
            const parsedOutput = JSON.parse(stdoutData.trim());
            if (parsedOutput.error) {
                reject(new Error(`Python Error: ${parsedOutput.error}`));
            } else {
                resolve(parsedOutput);
            }
          } catch (err) {
            reject(new Error(`Failed to parse Python output: ${err.message}. Stdout: ${stdoutData.trim()}`));
          }
        });
      });
    };
  }

  // Define browser-compatible and Node-compatible metadata for rendering in dashboard
  const modelData = {
    tag: "XGBoost",
    metrics: {
      accuracy: "0.6232",
      precision: "0.6882",
      recall: "0.6232",
      f1: "0.6438",
      strength: "Highly robust ensemble that handles outliers and captures complex non-linear interactions.",
      weakness: "Suffers on smaller/imbalanced classes like Distressed due to low sample size."
    },
    matrix: [
      [66, 28, 1, 4],
      [31, 83, 17, 3],
      [6, 49, 103, 14],
      [0, 0, 0, 1]
    ],
    labels: ["Investment-High", "Investment-Low", "Speculative", "Distressed"],
    shap: [
      ["Debt ratio (Sector Z-Score)", 95],
      ["Debt-to-equity ratio (Sector Z-Score)", 92],
      ["Cashflow debt coverage", 88],
      ["Leverage coverage (Sector Z-Score)", 84],
      ["Return on capital employed", 78],
      ["Equity multiplier (Sector Z-Score)", 72]
    ],
    shapStory: {
      positive: ["Debt ratio (Sector Z-Score)", "Debt-to-equity ratio (Sector Z-Score)", "Cashflow debt coverage"],
      negative: ["Return on capital employed", "Equity multiplier (Sector Z-Score)", "Leverage coverage (Sector Z-Score)"]
    }
  };

  const moduleExport = {
    predictXGBoost,
    name: "XGBoost",
    route: "/predict/xgboost",
    scriptPath: "python_models/xgboost_stuff/predict_xgboost.py",
    data: modelData
  };

  // Export for Node.js
  if (typeof module !== "undefined" && module.exports) {
    module.exports = moduleExport;
  }

  // Register globally for the Browser
  if (root) {
    root.MODEL_LIBRARY = root.MODEL_LIBRARY || {};
    root.MODEL_LIBRARY["XGBoost"] = modelData;
  }
})(typeof window !== "undefined" ? window : null);
