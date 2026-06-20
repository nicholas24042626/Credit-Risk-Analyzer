const { spawn } = require("node:child_process");
const path = require("node:path");

/**
 * Predict credit risk using the XGBoost Python model.
 * 
 * @param {Object} inputData - JSON object containing the financial ratios and Sector.
 * @returns {Promise<Object>} - Promise resolving to { prediction: "...", probabilities: {...} }
 */
function predictXGBoost(inputData) {
  return new Promise((resolve, reject) => {
    // Path to the companion Python script
    const scriptPath = path.join(__dirname, "predict_xgboost.py");
    
    // Use the virtual environment Python interpreter if available
    const fs = require("node:fs");
    const venvPath = path.join(__dirname, ".venv", "Scripts", "python.exe");
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
}

module.exports = { predictXGBoost };
