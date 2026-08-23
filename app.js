import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

const CAPTURE_INTERVAL_MS = 800;
const CAPTURE_SIZE = 256;

const modelStatusText = document.getElementById('modelStatusText');
const modelProgressBar = document.getElementById('modelProgressBar');

const tabCameraBtn = document.getElementById('tabCameraBtn');
const tabUploadBtn = document.getElementById('tabUploadBtn');
const cameraPanel = document.getElementById('cameraPanel');
const uploadPanel = document.getElementById('uploadPanel');

const video = document.getElementById('video');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const liveBadge = document.getElementById('liveBadge');
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const switchCameraBtn = document.getElementById('switchCameraBtn');
const cameraError = document.getElementById('cameraError');
const liveResults = document.getElementById('liveResults');

const dropZone = document.getElementById('dropZone');
const dropZoneText = document.getElementById('dropZoneText');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');
const uploadResults = document.getElementById('uploadResults');

const captureCanvas = document.getElementById('captureCanvas');
const captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });
captureCanvas.width = CAPTURE_SIZE;
captureCanvas.height = CAPTURE_SIZE;

let classifier = null;
let stream = null;
let facingMode = 'environment';
let liveRunning = false;
let liveTimer = null;

// ---------- Model loading ----------

const seenFiles = new Set();

(async () => {
  try {
    classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224', {
      progress_callback: onModelProgress,
    });
    modelStatusText.textContent = 'Model ready.';
    modelProgressBar.style.width = '100%';
    startCameraBtn.disabled = false;
    fileInput.disabled = false;
    dropZone.classList.remove('disabled');
  } catch (err) {
    console.error(err);
    modelStatusText.textContent = 'Failed to load model. Check your connection and reload the page.';
  }
})();

function onModelProgress(data) {
  if (data.status === 'progress') {
    seenFiles.add(data.file);
    const pct = Math.round(data.progress || 0);
    modelStatusText.textContent = `Downloading model… ${pct}% (${data.file})`;
    modelProgressBar.style.width = `${pct}%`;
  } else if (data.status === 'done') {
    modelStatusText.textContent = `Preparing model…`;
  }
}

// ---------- Tabs ----------

tabCameraBtn.addEventListener('click', () => switchTab('camera'));
tabUploadBtn.addEventListener('click', () => switchTab('upload'));

function switchTab(which) {
  const isCamera = which === 'camera';
  tabCameraBtn.classList.toggle('active', isCamera);
  tabUploadBtn.classList.toggle('active', !isCamera);
  tabCameraBtn.setAttribute('aria-selected', String(isCamera));
  tabUploadBtn.setAttribute('aria-selected', String(!isCamera));
  cameraPanel.classList.toggle('hidden', !isCamera);
  uploadPanel.classList.toggle('hidden', isCamera);
  if (!isCamera) {
    stopCamera();
  }
}

// ---------- Camera ----------

startCameraBtn.addEventListener('click', startCamera);
stopCameraBtn.addEventListener('click', stopCamera);
switchCameraBtn.addEventListener('click', async () => {
  facingMode = facingMode === 'environment' ? 'user' : 'environment';
  if (stream) {
    stopCamera({ keepUiRunning: true });
    await startCamera();
  }
});

async function startCamera() {
  hideCameraError();

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showCameraError('Camera access requires a secure context (HTTPS or localhost) and a supported browser.');
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    video.classList.add('active');
    cameraPlaceholder.classList.add('hidden');
    liveBadge.classList.remove('hidden');
    startCameraBtn.classList.add('hidden');
    stopCameraBtn.classList.remove('hidden');
    switchCameraBtn.classList.remove('hidden');

    liveRunning = true;
    scheduleNextCapture();
  } catch (err) {
    console.error(err);
    showCameraError(cameraErrorMessage(err));
  }
}

function stopCamera({ keepUiRunning = false } = {}) {
  liveRunning = false;
  if (liveTimer) {
    clearTimeout(liveTimer);
    liveTimer = null;
  }
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;

  if (!keepUiRunning) {
    video.classList.remove('active');
    cameraPlaceholder.classList.remove('hidden');
    liveBadge.classList.add('hidden');
    startCameraBtn.classList.remove('hidden');
    stopCameraBtn.classList.add('hidden');
    switchCameraBtn.classList.add('hidden');
    liveResults.innerHTML = '';
  }
}

function scheduleNextCapture() {
  if (!liveRunning) return;
  liveTimer = setTimeout(captureAndClassify, CAPTURE_INTERVAL_MS);
}

async function captureAndClassify() {
  if (!liveRunning || !classifier || video.readyState < 2) {
    scheduleNextCapture();
    return;
  }

  try {
    drawCenterCropSquare(video, captureCtx, CAPTURE_SIZE);
    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.85);
    const output = await classifier(dataUrl, { topk: 3 });
    renderResults(liveResults, output);
  } catch (err) {
    console.error(err);
  } finally {
    scheduleNextCapture();
  }
}

function drawCenterCropSquare(source, ctx, size) {
  const sw = source.videoWidth;
  const sh = source.videoHeight;
  const side = Math.min(sw, sh);
  const sx = (sw - side) / 2;
  const sy = (sh - side) / 2;
  ctx.drawImage(source, sx, sy, side, side, 0, 0, size, size);
}

function cameraErrorMessage(err) {
  if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
    return 'Camera permission was denied. Allow camera access in your browser settings and try again.';
  }
  if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
    return 'No camera was found on this device.';
  }
  if (err.name === 'NotReadableError') {
    return 'The camera is already in use by another application.';
  }
  return `Could not access the camera: ${err.message || err.name}`;
}

function showCameraError(message) {
  cameraError.textContent = message;
  cameraError.classList.remove('hidden');
}

function hideCameraError() {
  cameraError.classList.add('hidden');
}

// ---------- Upload ----------

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) classifyUploadedFile(file);
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) classifyUploadedFile(file);
});

async function classifyUploadedFile(file) {
  if (!classifier) return;

  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  uploadResults.innerHTML = '';
  dropZoneText.textContent = 'Classifying…';

  try {
    const output = await classifier(preview.src, { topk: 3 });
    renderResults(uploadResults, output);
    dropZoneText.textContent = 'Click to choose a different image, or drag one here';
  } catch (err) {
    console.error(err);
    dropZoneText.textContent = 'Something went wrong classifying that image. Try another.';
  }
}

// ---------- Shared rendering ----------

function renderResults(container, output) {
  container.innerHTML = '';
  for (const { label, score } of output) {
    const row = document.createElement('div');
    row.className = 'result-row';

    const labelRow = document.createElement('div');
    labelRow.className = 'result-label';

    const name = document.createElement('span');
    name.className = 'result-name';
    name.textContent = label.split(',')[0];

    const scoreEl = document.createElement('span');
    scoreEl.className = 'result-score';
    scoreEl.textContent = `${(score * 100).toFixed(1)}%`;

    labelRow.appendChild(name);
    labelRow.appendChild(scoreEl);

    const barTrack = document.createElement('div');
    barTrack.className = 'result-bar-track';
    const barFill = document.createElement('div');
    barFill.className = 'result-bar-fill';
    barFill.style.width = `${Math.round(score * 100)}%`;
    barTrack.appendChild(barFill);

    row.appendChild(labelRow);
    row.appendChild(barTrack);
    container.appendChild(row);
  }
}
