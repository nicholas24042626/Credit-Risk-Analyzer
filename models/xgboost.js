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
        const pythonProcess = spawn(pythonExecutable, [scriptPath], { windowsHide: true });
        
        if (pythonProcess.stdin) {
            pythonProcess.stdin.write(JSON.stringify(inputData));
            pythonProcess.stdin.end();
        }
        
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
    trained: false,
    metrics: null,
    labels: [],
    matrix: [],
    shap: [],
    shapStory: null
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
