const storageKey = "hw2-cnn-shape-model-v6";
const samplesKey = "hw2-cnn-user-samples-v6";

const canvas = document.querySelector("#drawCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const selectedButtons = document.querySelectorAll("[data-label]");
const elements = {
  layers: document.querySelector("#layersInput"),
  filters: document.querySelector("#filtersInput"),
  kernel: document.querySelector("#kernelInput"),
  learningRate: document.querySelector("#learningRateInput"),
  epochs: document.querySelector("#epochsInput"),
  prediction: document.querySelector("#predictionName"),
  status: document.querySelector("#statusText"),
  epoch: document.querySelector("#epochText"),
  loss: document.querySelector("#lossText"),
  accuracy: document.querySelector("#accuracyText"),
  samples: document.querySelector("#sampleCount"),
  storage: document.querySelector("#storageText"),
  weightsBadge: document.querySelector("#weightsBadge"),
  weightsSource: document.querySelector("#weightsSource"),
  weightsShape: document.querySelector("#weightsShape"),
  weightsSize: document.querySelector("#weightsSize"),
  loadWeights: document.querySelector("#loadWeightsBtn"),
  exportModel: document.querySelector("#exportBtn"),
  build: document.querySelector("#buildBtn"),
  reset: document.querySelector("#resetBtn"),
  predict: document.querySelector("#predictBtn"),
  addSample: document.querySelector("#addSampleBtn")
};

let model = null;
let selectedLabel = "circle";
let userSamples = [];
let drawing = false;

function setStatus(text) {
  elements.status.textContent = text;
}

function clearCanvas() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const source = event.touches ? event.touches[0] : event;
  return {
    x: (source.clientX - rect.left) * (canvas.width / rect.width),
    y: (source.clientY - rect.top) * (canvas.height / rect.height)
  };
}

function startDraw(event) {
  drawing = true;
  const point = canvasPoint(event);
  ctx.beginPath();
  ctx.moveTo(point.x, point.y);
  event.preventDefault();
}

function draw(event) {
  if (!drawing) return;
  const point = canvasPoint(event);
  ctx.lineTo(point.x, point.y);
  ctx.strokeStyle = "black";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.stroke();
  event.preventDefault();
}

function stopDraw() {
  drawing = false;
}

function readOptions() {
  return {
    layers: Number(elements.layers.value),
    filters: Number(elements.filters.value),
    kernel: Number(elements.kernel.value),
    learningRate: Number(elements.learningRate.value),
    seed: 2026
  };
}

function lockOptions(locked) {
  elements.layers.disabled = locked;
  elements.filters.disabled = locked;
  elements.kernel.disabled = locked;
  elements.learningRate.disabled = locked;
}

function saveModel() {
  localStorage.setItem(storageKey, JSON.stringify(model));
  elements.storage.textContent = "נשמר";
  refreshWeightsCard("LocalStorage");
}

function applyLoadedModel(loadedModel, storageText) {
  if (loadedModel.version !== 6 || loadedModel.options.inputSize !== ShapeCNN.inputSize) {
    return false;
  }
  model = loadedModel;
  elements.layers.value = model.options.layers;
  elements.filters.value = model.options.filters;
  elements.kernel.value = model.options.kernel;
  elements.learningRate.value = model.options.learningRate;
  lockOptions(true);
  elements.storage.textContent = storageText;
  refreshWeightsCard(storageText);
  return true;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function modelShapeText(savedModel) {
  if (!savedModel) return "-";
  return `${savedModel.options.layers} layers, ${savedModel.options.filters} filters, ${savedModel.options.kernel}x${savedModel.options.kernel}, dense ${savedModel.options.denseInputs}`;
}

function refreshWeightsCard(source) {
  const raw = localStorage.getItem(storageKey);
  if (!raw) {
    elements.weightsBadge.textContent = "אין שמירה";
    elements.weightsBadge.classList.remove("ready");
    elements.weightsSource.textContent = "-";
    elements.weightsShape.textContent = "-";
    elements.weightsSize.textContent = "-";
    return;
  }

  try {
    const savedModel = JSON.parse(raw);
    elements.weightsBadge.textContent = "נשמר";
    elements.weightsBadge.classList.add("ready");
    elements.weightsSource.textContent = source || "LocalStorage";
    elements.weightsShape.textContent = modelShapeText(savedModel);
    elements.weightsSize.textContent = formatBytes(new Blob([raw]).size);
  } catch (error) {
    elements.weightsBadge.textContent = "שגיאה";
    elements.weightsBadge.classList.remove("ready");
    elements.weightsSource.textContent = "Invalid JSON";
    elements.weightsShape.textContent = "-";
    elements.weightsSize.textContent = "-";
  }
}

function loadModel() {
  const raw = localStorage.getItem(storageKey);
  if (!raw) return false;
  if (!applyLoadedModel(JSON.parse(raw), "נטען")) {
    localStorage.removeItem(storageKey);
    model = null;
    return false;
  }
  return true;
}

function loadWeightsFromStorage() {
  if (loadModel()) {
    setStatus("Weights loaded from LocalStorage");
    return;
  }
  refreshWeightsCard();
  setStatus("No saved weights found");
}

async function loadPretrainedModel() {
  try {
    const response = await fetch("trained-model.json", { cache: "no-store" });
    if (!response.ok) return false;
    const pretrainedModel = await response.json();
    if (!applyLoadedModel(pretrainedModel, "Pretrained")) return false;
    saveModel();
    elements.storage.textContent = "Pretrained";
    return true;
  } catch (error) {
    console.log("No pretrained model file loaded", error);
    return false;
  }
}

function exportTrainedModel() {
  if (!model) {
    setStatus("No trained model found");
    return;
  }
  const blob = new Blob([JSON.stringify(model, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "trained-model.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  setStatus("Exported trained-model.json");
}

function saveSamples() {
  localStorage.setItem(samplesKey, JSON.stringify(userSamples));
  elements.samples.textContent = String(userSamples.length);
}

function loadSamples() {
  userSamples = JSON.parse(localStorage.getItem(samplesKey) || "[]");
  elements.samples.textContent = String(userSamples.length);
}

function buildModel() {
  model = ShapeCNN.makeModel(readOptions());
  lockOptions(true);
  saveModel();
  setStatus("מודל חדש נוצר והפרמטרים קובעו");
}

function updatePrediction(probabilities) {
  const names = ["Circle", "Square", "Triangle"];
  const ids = ["Circle", "Square", "Triangle"];
  let best = 0;
  probabilities.forEach((value, index) => {
    if (value > probabilities[best]) best = index;
    document.querySelector(`#bar${ids[index]}`).value = value;
    document.querySelector(`#pct${ids[index]}`).textContent = `${Math.round(value * 100)}%`;
  });
  const label = ShapeCNN.labels[best];
  elements.prediction.textContent = `${ShapeCNN.hebrewLabels[label]} (${names[best]})`;
}

function predictCurrent() {
  if (!model) {
    model = ShapeCNN.makeModel(readOptions());
    lockOptions(true);
    elements.storage.textContent = "מודל זמני";
    elements.prediction.textContent = "מודל לא מאומן";
    setStatus("No trained model found - using temporary untrained model");
  }
  const matrix = ShapeCNN.canvasToMatrix(canvas);
  const result = ShapeCNN.forward(model, matrix);
  console.log("Prediction detail", {
    probabilities: result.probabilities.map(value => Number(value.toFixed(4))),
    learnedLogits: (result.learnedLogits || []).map(value => Number(value.toFixed(4))),
    shapeLogits: (result.shapeLogits || []).map(value => Number(value.toFixed(4))),
    descriptor: (result.descriptor || []).map(value => Number(value.toFixed(4)))
  });
  updatePrediction(result.probabilities);
  setStatus("ניבוי הושלם");
}

async function addUserSample() {
  if (!model) buildModel();
  const matrix = ShapeCNN.canvasToMatrix(canvas);
  userSamples.push({
    label: selectedLabel,
    input: matrix
  });
  saveSamples();
  await trainManualSamples(`נוספה דגימת ${ShapeCNN.hebrewLabels[selectedLabel]}`);
}

function labelIndex(label) {
  return ShapeCNN.labels.indexOf(label);
}

function summarizeDataset(data, title) {
  const summary = {};
  for (const label of ShapeCNN.labels) {
    const classSamples = data.filter(sample => sample.label === label);
    summary[label] = {
      total: classSamples.length,
      target: ShapeCNN.labelVectors[label],
      first3Labels: classSamples.slice(0, 3).map(sample => sample.label)
    };
  }
  const labelIndicesPerSample = data.map((sample, sampleIndex) => ({
    sample: sampleIndex,
    label: sample.label,
    labelIndex: labelIndex(sample.label)
  }));
  console.log(`${title} dataset summary`, summary);
  console.log(`${title} labels count`, {
    circle: summary.circle.total,
    square: summary.square.total,
    triangle: summary.triangle.total
  });
  console.log(`${title} labelIndex per sample`, labelIndicesPerSample);
  console.log("Expected one-hot labels", {
    circle: [1, 0, 0],
    square: [0, 1, 0],
    triangle: [0, 0, 1]
  });
}

function cleanShapePredictions() {
  const results = {};
  for (const label of ShapeCNN.labels) {
    const result = ShapeCNN.forward(model, ShapeCNN.generateCleanShape(label));
    results[label] = {
      probabilities: result.probabilities.map(value => Number(value.toFixed(4))),
      predicted: ShapeCNN.labels[result.probabilities.indexOf(Math.max(...result.probabilities))]
    };
  }
  return results;
}

function runCleanPredictionTest() {
  const results = cleanShapePredictions();
  console.log("Post-training clean shape prediction test", results);
}

function setBusy(isBusy) {
  elements.build.disabled = isBusy;
  elements.exportModel.disabled = isBusy;
  elements.loadWeights.disabled = isBusy;
  elements.reset.disabled = isBusy;
  elements.addSample.disabled = isBusy;
}

async function trainManualSamples(prefix) {
  if (!userSamples.length) {
    setStatus("אין דגימות ידניות לאימון");
    return;
  }
  setBusy(true);
  const epochs = Number(elements.epochs.value);
  summarizeDataset(userSamples, "Manual training");

  for (let epoch = 1; epoch <= epochs; epoch++) {
    const data = ShapeCNN.shuffle(userSamples.slice(), ShapeCNN.createRandom(5000 + epoch + userSamples.length));
    let loss = 0;
    let correct = 0;

    for (const sample of data) {
      const result = ShapeCNN.trainOne(model, sample.input, labelIndex(sample.label));
      loss += result.loss;
      correct += result.correct ? 1 : 0;
    }

    elements.loss.textContent = (loss / data.length).toFixed(4);
    elements.accuracy.textContent = `${Math.round((correct / data.length) * 100)}%`;
    elements.epoch.textContent = `${epoch} / ${epochs}`;
    console.log(`Manual training epoch ${epoch} prediction probabilities`, cleanShapePredictions());
    setStatus(`${prefix}: אימון ידני ${epoch} מתוך ${epochs}`);
    await new Promise(resolve => setTimeout(resolve, 10));
  }

  saveModel();
  setBusy(false);
  setStatus("האימון הידני נשמר ב-LocalStorage");
  runCleanPredictionTest();
  predictCurrent();
}

async function bootstrapIfNeeded() {
  if (loadModel()) {
    setStatus("Local model loaded");
    return;
  }
  if (await loadPretrainedModel()) {
    setStatus("Pretrained model loaded");
    return;
  }
  model = null;
  lockOptions(false);
  elements.storage.textContent = "לא נמצא";
  refreshWeightsCard();
  setStatus("No trained model found");
}

function resetAll() {
  localStorage.removeItem(storageKey);
  localStorage.removeItem(samplesKey);
  localStorage.removeItem("hw2-cnn-shape-model-v3");
  localStorage.removeItem("hw2-cnn-user-samples-v3");
  localStorage.removeItem("hw2-cnn-shape-model-v4");
  localStorage.removeItem("hw2-cnn-user-samples-v4");
  localStorage.removeItem("hw2-cnn-shape-model-v5");
  localStorage.removeItem("hw2-cnn-user-samples-v5");
  localStorage.removeItem("hw2-cnn-shape-model-v2");
  localStorage.removeItem("hw2-cnn-user-samples-v2");
  userSamples = [];
  elements.samples.textContent = "0";
  lockOptions(false);
  model = null;
  elements.epoch.textContent = "-";
  elements.loss.textContent = "-";
  elements.accuracy.textContent = "-";
  elements.storage.textContent = "נמחק";
  refreshWeightsCard();
  clearCanvas();
  setStatus("המודל אופס");
}

selectedButtons.forEach(button => {
  button.addEventListener("click", () => {
    selectedLabel = button.dataset.label;
    selectedButtons.forEach(item => item.classList.toggle("selected", item === button));
  });
});

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", stopDraw);
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
window.addEventListener("touchend", stopDraw);

document.querySelector("#clearCanvas").addEventListener("click", clearCanvas);
elements.predict.addEventListener("click", predictCurrent);
elements.addSample.addEventListener("click", addUserSample);
elements.build.addEventListener("click", buildModel);
elements.exportModel.addEventListener("click", exportTrainedModel);
elements.loadWeights.addEventListener("click", loadWeightsFromStorage);
elements.reset.addEventListener("click", resetAll);

clearCanvas();
selectedButtons[0].classList.add("selected");
loadSamples();
refreshWeightsCard();
bootstrapIfNeeded();
