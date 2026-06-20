const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");

const randomForestModel = require("./models/randomForest");

const root = __dirname;
const port = Number(process.env.PORT || 3000);

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon"
};

function send(res, statusCode, body, contentType) {
  res.writeHead(statusCode, { "Content-Type": contentType });
  res.end(body);
}

function sendJson(res, statusCode, payload) {
  send(res, statusCode, JSON.stringify(payload), "application/json; charset=utf-8");
}

function safeResolve(requestPath) {
  const cleanPath = requestPath.split("?")[0].split("#")[0];
  const relativePath = cleanPath === "/" ? "/views/index.html" : cleanPath;
  const resolved = path.resolve(root, `.${relativePath}`);
  const rootWithSep = root.endsWith(path.sep) ? root : `${root}${path.sep}`;
  if (
    !resolved.startsWith(rootWithSep) &&
    resolved !== path.resolve(root, "views/index.html")
  ) {
    return null;
  }
  return resolved;
}

const server = http.createServer((req, res) => {
  if (!req.url) {
    send(res, 400, "Bad Request", "text/plain; charset=utf-8");
    return;
  }

  // Predict endpoint for the Python-trained Random Forest model.
  if (req.method === "POST" && req.url === randomForestModel.route) {
    const scriptPath = path.join(__dirname, "python_models", "predict_random_forest.py");

    if (!fs.existsSync(scriptPath)) {
      sendJson(res, 500, {
        error: "Python prediction script not found.",
        expectedPath: scriptPath
      });
      return;
    }

    let body = "";

    req.on("data", (chunk) => {
      body += chunk;
    });

    req.on("end", () => {
      let requestPayload;

      try {
        requestPayload = body ? JSON.parse(body) : {};
      } catch (err) {
        sendJson(res, 400, {
          error: "Invalid JSON request body.",
          details: err.message
        });
        return;
      }

      const pythonArgs = [scriptPath, JSON.stringify(requestPayload)];
      const pythonProcess = spawn("python", pythonArgs, { windowsHide: true });
      let stdoutData = "";
      let stderrData = "";
      let hasResponded = false;

      pythonProcess.stdout.on("data", (data) => {
        stdoutData += data.toString();
      });

      pythonProcess.stderr.on("data", (data) => {
        stderrData += data.toString();
      });

      pythonProcess.on("error", (err) => {
        if (hasResponded) {
          return;
        }
        hasResponded = true;
        sendJson(res, 500, {
          error: "Failed to start Python process.",
          details: err.message
        });
      });

      pythonProcess.on("close", (code) => {
        if (hasResponded) {
          return;
        }

        if (code !== 0) {
          hasResponded = true;
          sendJson(res, 500, {
            error: "Python prediction process failed.",
            exitCode: code,
            stderr: stderrData.trim()
          });
          return;
        }

        const trimmedOutput = stdoutData.trim();

        try {
          const parsedOutput = JSON.parse(trimmedOutput);
          hasResponded = true;
          sendJson(res, 200, parsedOutput);
        } catch (err) {
          hasResponded = true;
          sendJson(res, 500, {
            error: "Python process returned invalid JSON.",
            details: err.message,
            stdout: trimmedOutput,
            stderr: stderrData.trim()
          });
        }
      });
    });

    return;
  }

  const filePath = safeResolve(req.url);
  if (!filePath) {
    send(res, 403, "Forbidden", "text/plain; charset=utf-8");
    return;
  }

  fs.readFile(filePath, (err, data) => {
    if (err) {
      if (err.code === "ENOENT") {
        send(res, 404, "Not Found", "text/plain; charset=utf-8");
      } else {
        send(res, 500, "Internal Server Error", "text/plain; charset=utf-8");
      }
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType = mimeTypes[ext] || "application/octet-stream";
    res.writeHead(200, { "Content-Type": contentType });
    res.end(data);
  });
});

server.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
