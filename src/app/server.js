const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawn } = require('child_process');

const MODEL_PATH = process.env.MODEL_PATH || path.join(__dirname, '..', '..', 'models', 'best_auc');
const VENV_PYTHON = path.join(__dirname, '..', '..', 'venv', 'bin', 'python');
const DEFAULT_PYTHON_CMD = fs.existsSync(VENV_PYTHON) ? VENV_PYTHON : 'python';
const PYTHON_CMD = process.env.PYTHON_CMD || DEFAULT_PYTHON_CMD;
const MODEL_SCRIPT = process.env.MODEL_SCRIPT || path.join(__dirname, '..', 'model.py');
const PORT = process.env.PORT || 3000;

function sendJson(res, status, data) {
  const payload = JSON.stringify(data);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(payload),
  });
  res.end(payload);
}

function handlePredict(req, res) {
  if (req.method !== 'POST') {
    return sendJson(res, 405, { error: 'Method not allowed' });
  }

  const contentType = (req.headers['content-type'] || '').toLowerCase();
  if (!contentType.includes('application/json')) {
    return sendJson(res, 415, { error: 'Content-Type must be application/json' });
  }

  let body = '';
  req.on('data', (chunk) => {
    body += chunk;
  });

  req.on('end', () => {
    let payload;
    try {
      payload = JSON.parse(body);
    } catch (err) {
      return sendJson(res, 400, { error: 'Invalid JSON body' });
    }

    if (!payload.image_base64) {
      return sendJson(res, 400, { error: 'Missing image_base64 field' });
    }

    const imageBuffer = Buffer.from(payload.image_base64, 'base64');
    const tempPath = path.join(os.tmpdir(), `pneumonia-${Date.now()}-${Math.random().toString(36).slice(2)}.png`);

    fs.writeFile(tempPath, imageBuffer, (writeErr) => {
      if (writeErr) {
        return sendJson(res, 500, { error: 'Failed to save uploaded image' });
      }

      const args = [MODEL_SCRIPT, '--image', tempPath, '--model', MODEL_PATH, '--json'];
      const py = spawn(PYTHON_CMD, args);
      let stdout = '';
      let stderr = '';

      py.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      py.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      py.on('close', (code) => {
        fs.unlink(tempPath, () => {});
        if (code !== 0) {
          console.error('Prediction process failed:', stderr.trim());
          return sendJson(res, 500, {
            error: 'Prediction failed',
            details: stderr.trim() || stdout.trim(),
          });
        }

        try {
          const result = JSON.parse(stdout);
          return sendJson(res, 200, result);
        } catch (err) {
          console.error('Failed to parse prediction output:', stdout.trim());
          return sendJson(res, 500, { error: 'Invalid prediction output', details: stdout.trim() });
        }
      });
    });
  });
}

function sendFile(res, filePath, contentType) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      return sendJson(res, 500, { error: 'Failed to read file' });
    }
    res.writeHead(200, { 'Content-Type': contentType, 'Content-Length': data.length });
    res.end(data);
  });
}

const server = http.createServer((req, res) => {
  if (req.url === '/predict') {
    return handlePredict(req, res);
  }
  if (req.url === '/') {
    const indexPath = path.join(__dirname, '..', 'app', 'index.html');
    return sendFile(res, indexPath, 'text/html');
  }
  if (req.url === '/style.css') {
    const cssPath = path.join(__dirname, '..', 'app', 'style.css');
    return sendFile(res, cssPath, 'text/css');
  }
  sendJson(res, 404, { error: 'Not found' });
});

server.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
