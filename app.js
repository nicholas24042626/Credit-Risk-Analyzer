const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");

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
