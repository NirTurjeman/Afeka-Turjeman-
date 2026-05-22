const ShapeCNN = (() => {
  const labels = ["circle", "square", "triangle"];
  const hebrewLabels = {
    circle: "עיגול",
    square: "ריבוע",
    triangle: "משולש"
  };
  const labelVectors = {
    circle: [1, 0, 0],
    square: [0, 1, 0],
    triangle: [0, 0, 1]
  };
  const inputSize = 28;
  const descriptorSize = 15;

  function createRandom(seed) {
    let state = seed >>> 0;
    return () => {
      state = (1664525 * state + 1013904223) >>> 0;
      return state / 4294967296;
    };
  }

  function zeros(channels, height, width) {
    return Array.from({ length: channels }, () =>
      Array.from({ length: height }, () => Array(width).fill(0))
    );
  }

  function outputSizeAfterPool(size) {
    return Math.floor(size / 2);
  }

  function makeModel(options) {
    const rand = createRandom(options.seed || 1234);
    const layers = [];
    let inChannels = 1;
    let height = inputSize;
    let width = inputSize;

    for (let layerIndex = 0; layerIndex < options.layers; layerIndex++) {
      const outChannels = options.filters;
      const kernel = options.kernel;
      const scale = Math.sqrt(2 / (inChannels * kernel * kernel));
      const weights = Array.from({ length: outChannels }, () =>
        Array.from({ length: inChannels }, () =>
          Array.from({ length: kernel }, () =>
            Array.from({ length: kernel }, () => (rand() * 2 - 1) * scale)
          )
        )
      );
      layers.push({
        inChannels,
        outChannels,
        kernel,
        weights,
        biases: Array(outChannels).fill(0)
      });
      inChannels = outChannels;
      height = outputSizeAfterPool(height);
      width = outputSizeAfterPool(width);
    }

    const cnnInputs = inChannels * height * width;
    const denseInputs = cnnInputs + descriptorSize;
    const denseScale = Math.sqrt(2 / denseInputs);
    const dense = {
      weights: Array.from({ length: 3 }, () =>
        Array.from({ length: denseInputs }, () => (rand() * 2 - 1) * denseScale)
      ),
      biases: Array(3).fill(0)
    };

    return {
      version: 6,
      labels,
      labelVectors,
      options: {
        layers: options.layers,
        filters: options.filters,
        kernel: options.kernel,
        learningRate: options.learningRate,
        inputSize,
        denseInputs,
        cnnInputs,
        descriptorSize
      },
      layers,
      dense
    };
  }

  function convForward(layer, input) {
    const outHeight = input[0].length;
    const outWidth = input[0][0].length;
    const pad = Math.floor(layer.kernel / 2);
    const pre = zeros(layer.outChannels, outHeight, outWidth);
    const output = zeros(layer.outChannels, outHeight, outWidth);

    for (let oc = 0; oc < layer.outChannels; oc++) {
      for (let y = 0; y < outHeight; y++) {
        for (let x = 0; x < outWidth; x++) {
          let sum = layer.biases[oc];
          for (let ic = 0; ic < layer.inChannels; ic++) {
            for (let ky = 0; ky < layer.kernel; ky++) {
              for (let kx = 0; kx < layer.kernel; kx++) {
                const inputY = y + ky - pad;
                const inputX = x + kx - pad;
                if (inputY >= 0 && inputY < input[ic].length && inputX >= 0 && inputX < input[ic][0].length) {
                  sum += input[ic][inputY][inputX] * layer.weights[oc][ic][ky][kx];
                }
              }
            }
          }
          pre[oc][y][x] = sum;
          output[oc][y][x] = Math.max(0, sum);
        }
      }
    }
    return { pre, output };
  }

  function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(value => Math.exp(value - maxLogit));
    const total = exps.reduce((sum, value) => sum + value, 0);
    return exps.map(value => value / Math.max(total, 1e-12));
  }

  function maxPool2x2(activation) {
    const channels = activation.length;
    const outHeight = Math.floor(activation[0].length / 2);
    const outWidth = Math.floor(activation[0][0].length / 2);
    const output = zeros(channels, outHeight, outWidth);
    const switches = zeros(channels, outHeight, outWidth);

    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < outHeight; y++) {
        for (let x = 0; x < outWidth; x++) {
          let maxValue = -Infinity;
          let maxIndex = 0;
          for (let py = 0; py < 2; py++) {
            for (let px = 0; px < 2; px++) {
              const inputY = y * 2 + py;
              const inputX = x * 2 + px;
              const value = activation[c][inputY][inputX];
              if (value > maxValue) {
                maxValue = value;
                maxIndex = py * 2 + px;
              }
            }
          }
          output[c][y][x] = maxValue;
          switches[c][y][x] = maxIndex;
        }
      }
    }
    return { output, switches };
  }

  function flatten(activation) {
    const features = [];
    for (let c = 0; c < activation.length; c++) {
      for (let y = 0; y < activation[c].length; y++) {
        for (let x = 0; x < activation[c][0].length; x++) {
          features.push(activation[c][y][x]);
        }
      }
    }
    return features;
  }

  function unflatten(features, shape) {
    const [channels, height, width] = shape;
    const output = zeros(channels, height, width);
    let index = 0;
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          output[c][y][x] = features[index++];
        }
      }
    }
    return output;
  }

  function cellAverage(matrix, yStart, yEnd, xStart, xEnd) {
    let sum = 0;
    let count = 0;
    for (let y = yStart; y < yEnd; y++) {
      for (let x = xStart; x < xEnd; x++) {
        sum += matrix[y][x];
        count++;
      }
    }
    return count ? sum / count : 0;
  }

  function shapeDescriptor(inputMatrix) {
    const features = [];
    const rows = inputMatrix.length;
    const cols = inputMatrix[0].length;
    let ink = 0;
    let edgeInk = 0;
    let minX = cols;
    let minY = rows;
    let maxX = -1;
    let maxY = -1;

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const value = inputMatrix[y][x];
        ink += value;
        if (value > 0.18) {
          minX = Math.min(minX, x);
          minY = Math.min(minY, y);
          maxX = Math.max(maxX, x);
          maxY = Math.max(maxY, y);
        }
        if (y === 0 || x === 0 || y === rows - 1 || x === cols - 1) {
          edgeInk += value;
        }
      }
    }

    for (let cy = 0; cy < 3; cy++) {
      for (let cx = 0; cx < 3; cx++) {
        features.push(cellAverage(
          inputMatrix,
          Math.floor((cy * rows) / 3),
          Math.floor(((cy + 1) * rows) / 3),
          Math.floor((cx * cols) / 3),
          Math.floor(((cx + 1) * cols) / 3)
        ));
      }
    }

    const hasInk = maxX >= minX && maxY >= minY;
    const width = hasInk ? maxX - minX + 1 : 1;
    const height = hasInk ? maxY - minY + 1 : 1;
    const boxArea = width * height;
    const totalArea = rows * cols;
    const center = features[4];
    const topCorners = features[0] + features[2];
    const bottomCorners = features[6] + features[8];

    features.push(ink / totalArea);
    features.push(edgeInk / Math.max(1, rows * 2 + cols * 2 - 4));
    features.push(width / cols);
    features.push(height / rows);
    features.push(ink / Math.max(1, boxArea));
    features.push((topCorners + bottomCorners) / Math.max(0.001, center + 0.05));
    return features;
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  function geometryLogits(descriptor) {
    const top = (descriptor[0] + descriptor[1] + descriptor[2]) / 3;
    const middle = (descriptor[3] + descriptor[4] + descriptor[5]) / 3;
    const bottom = (descriptor[6] + descriptor[7] + descriptor[8]) / 3;
    const left = (descriptor[0] + descriptor[3] + descriptor[6]) / 3;
    const right = (descriptor[2] + descriptor[5] + descriptor[8]) / 3;
    const corners = (descriptor[0] + descriptor[2] + descriptor[6] + descriptor[8]) / 4;
    const sideMiddles = (descriptor[1] + descriptor[3] + descriptor[5] + descriptor[7]) / 4;
    const center = descriptor[4];
    const widthRatio = descriptor[11];
    const heightRatio = descriptor[12];
    const fillRatio = descriptor[13];
    const aspectBalance = 1 - Math.abs(widthRatio - heightRatio);
    const hollow = clamp01((sideMiddles + corners) - center * 1.8);
    const frameBalance = 1 - Math.abs(top - bottom) - Math.abs(left - right);
    const bottomBias = clamp01(bottom - top);

    const squareScore =
      2.2 * clamp01(frameBalance) +
      2.0 * aspectBalance +
      2.0 * hollow +
      1.5 * corners -
      1.4 * bottomBias;

    const triangleScore =
      2.4 * bottomBias +
      1.2 * clamp01(bottom - middle * 0.5) +
      0.8 * clamp01(center - top) -
      0.8 * corners;

    const circleScore =
      1.5 * aspectBalance +
      1.1 * clamp01(sideMiddles - corners * 0.35) +
      0.8 * clamp01(fillRatio - 0.18) -
      0.7 * hollow;

    return [circleScore, squareScore, triangleScore].map(value => value * 0.9);
  }

  function forward(model, inputMatrix) {
    let activation = [inputMatrix];
    const cache = [];

    for (const layer of model.layers) {
      const conv = convForward(layer, activation);
      const pool = maxPool2x2(conv.output);
      cache.push({
        input: activation,
        pre: conv.pre,
        convOutput: conv.output,
        poolSwitches: pool.switches,
        pooledOutput: pool.output
      });
      activation = pool.output;
    }

    const cnnFeatures = flatten(activation);
    const descriptor = shapeDescriptor(inputMatrix);
    const features = cnnFeatures.concat(descriptor);

    const learnedLogits = model.dense.weights.map((row, classIndex) =>
      row.reduce((sum, weight, i) => sum + weight * features[i], model.dense.biases[classIndex])
    );
    const shapeLogits = geometryLogits(descriptor);
    const logits = learnedLogits.map((value, index) => value + shapeLogits[index]);

    return {
      probabilities: softmax(logits),
      features,
      cnnFeatures,
      descriptor,
      learnedLogits,
      shapeLogits,
      finalActivation: activation,
      cache
    };
  }

  function trainOne(model, inputMatrix, labelIndex) {
    const lr = model.options.learningRate;
    const pass = forward(model, inputMatrix);
    const target = [0, 0, 0];
    target[labelIndex] = 1;
    const error = pass.probabilities.map((prediction, index) => prediction - target[index]);
    const loss = -Math.log(Math.max(pass.probabilities[labelIndex], 1e-9));
    const predicted = pass.probabilities.indexOf(Math.max(...pass.probabilities));

    const oldDenseWeights = model.dense.weights.map(row => row.slice());
    const gradFeatures = Array(pass.features.length).fill(0);
    for (let classIndex = 0; classIndex < 3; classIndex++) {
      for (let i = 0; i < pass.features.length; i++) {
        gradFeatures[i] += error[classIndex] * oldDenseWeights[classIndex][i];
        model.dense.weights[classIndex][i] -= lr * error[classIndex] * pass.features[i];
      }
      model.dense.biases[classIndex] -= lr * error[classIndex];
    }

    const cnnInputs = model.options.cnnInputs || (pass.finalActivation.length * pass.finalActivation[0].length * pass.finalActivation[0][0].length);
    const gradCnnFeatures = gradFeatures.slice(0, cnnInputs);
    let gradActivation = unflatten(gradCnnFeatures, [
      pass.finalActivation.length,
      pass.finalActivation[0].length,
      pass.finalActivation[0][0].length
    ]);

    for (let layerIndex = model.layers.length - 1; layerIndex >= 0; layerIndex--) {
      const layer = model.layers[layerIndex];
      const cached = pass.cache[layerIndex];
      const gradConvOutput = zeros(
        layer.outChannels,
        cached.convOutput[0].length,
        cached.convOutput[0][0].length
      );
      for (let oc = 0; oc < gradActivation.length; oc++) {
        for (let y = 0; y < gradActivation[oc].length; y++) {
          for (let x = 0; x < gradActivation[oc][0].length; x++) {
            const maxIndex = cached.poolSwitches[oc][y][x];
            const inputY = y * 2 + Math.floor(maxIndex / 2);
            const inputX = x * 2 + (maxIndex % 2);
            gradConvOutput[oc][inputY][inputX] += gradActivation[oc][y][x];
          }
        }
      }
      const gradInput = zeros(layer.inChannels, cached.input[0].length, cached.input[0][0].length);
      const gradWeights = Array.from({ length: layer.outChannels }, () =>
        Array.from({ length: layer.inChannels }, () =>
          Array.from({ length: layer.kernel }, () => Array(layer.kernel).fill(0))
        )
      );
      const gradBiases = Array(layer.outChannels).fill(0);
      const pad = Math.floor(layer.kernel / 2);

      for (let oc = 0; oc < layer.outChannels; oc++) {
        for (let y = 0; y < gradConvOutput[oc].length; y++) {
          for (let x = 0; x < gradConvOutput[oc][0].length; x++) {
            const grad = cached.pre[oc][y][x] > 0 ? gradConvOutput[oc][y][x] : 0;
            gradBiases[oc] += grad;
            for (let ic = 0; ic < layer.inChannels; ic++) {
              for (let ky = 0; ky < layer.kernel; ky++) {
                for (let kx = 0; kx < layer.kernel; kx++) {
                  const inputY = y + ky - pad;
                  const inputX = x + kx - pad;
                  if (inputY >= 0 && inputY < cached.input[ic].length && inputX >= 0 && inputX < cached.input[ic][0].length) {
                    gradWeights[oc][ic][ky][kx] += cached.input[ic][inputY][inputX] * grad;
                    gradInput[ic][inputY][inputX] += layer.weights[oc][ic][ky][kx] * grad;
                  }
                }
              }
            }
          }
        }
      }

      for (let oc = 0; oc < layer.outChannels; oc++) {
        layer.biases[oc] -= lr * gradBiases[oc];
        for (let ic = 0; ic < layer.inChannels; ic++) {
          for (let ky = 0; ky < layer.kernel; ky++) {
            for (let kx = 0; kx < layer.kernel; kx++) {
              layer.weights[oc][ic][ky][kx] -= lr * gradWeights[oc][ic][ky][kx];
            }
          }
        }
      }
      gradActivation = gradInput;
    }

    return { loss, correct: predicted === labelIndex };
  }

  function generateShape(label, rand) {
    return renderShape(label, rand, false);
  }

  function generateCleanShape(label) {
    return renderShape(label, createRandom(777), true);
  }

  function renderShape(label, rand, clean) {
    const canvas = document.createElement("canvas");
    canvas.width = 96;
    canvas.height = 96;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, 96, 96);
    ctx.fillStyle = "black";
    ctx.strokeStyle = "black";
    ctx.lineWidth = clean ? 7 : 3 + rand() * 8;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    const cx = clean ? 48 : 48 + (rand() - 0.5) * 22;
    const cy = clean ? 48 : 48 + (rand() - 0.5) * 22;
    const size = clean ? 58 : 38 + rand() * 28;
    const rotation = clean ? 0 : (rand() - 0.5) * (label === "square" ? Math.PI * 0.18 : Math.PI * 0.55);
    const shouldFill = clean ? false : rand() > 0.42;

    if (label === "circle") {
      ctx.beginPath();
      ctx.ellipse(cx, cy, size / 2, clean ? size / 2 : size * (0.42 + rand() * 0.16), 0, 0, Math.PI * 2);
      if (shouldFill) ctx.fill();
      ctx.stroke();
    } else if (label === "square") {
      const offset = size / 2;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(rotation);
      if (shouldFill && rand() > 0.35) {
        ctx.fillRect(-offset, -offset, size, size);
      }
      ctx.beginPath();
      const wobble = clean ? 0 : size * 0.08;
      const p1 = [-offset + (rand() - 0.5) * wobble, -offset + (rand() - 0.5) * wobble];
      const p2 = [offset + (rand() - 0.5) * wobble, -offset + (rand() - 0.5) * wobble];
      const p3 = [offset + (rand() - 0.5) * wobble, offset + (rand() - 0.5) * wobble];
      const p4 = [-offset + (rand() - 0.5) * wobble, offset + (rand() - 0.5) * wobble];
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.lineTo(p3[0], p3[1]);
      ctx.lineTo(p4[0], p4[1]);
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
    } else {
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(rotation);
      ctx.beginPath();
      ctx.moveTo(0, -size / 2);
      ctx.lineTo(-size / 2, size / 2);
      ctx.lineTo(size / 2, size / 2);
      ctx.closePath();
      if (shouldFill) ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    const matrix = canvasToMatrix(canvas);
    return clean ? matrix : addNoise(matrix, rand);
  }

  function addNoise(matrix, rand) {
    return matrix.map(row => row.map(value => {
      const jitter = (rand() - 0.5) * 0.12;
      const speckle = rand() < 0.025 ? rand() * 0.7 : 0;
      return Math.max(0, Math.min(1, value + jitter + speckle));
    }));
  }

  function generateDataset(samplesPerClass, seed) {
    const rand = createRandom(seed);
    const data = [];
    for (const label of labels) {
      for (let i = 0; i < samplesPerClass; i++) {
        data.push({
          label,
          input: generateShape(label, rand)
        });
      }
    }
    shuffle(data, rand);
    return data;
  }

  function shuffle(data, rand) {
    for (let i = data.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [data[i], data[j]] = [data[j], data[i]];
    }
    return data;
  }

  function canvasToMatrix(canvas) {
    const sample = document.createElement("canvas");
    sample.width = inputSize;
    sample.height = inputSize;
    const ctx = sample.getContext("2d", { willReadFrequently: true });
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, inputSize, inputSize);
    ctx.drawImage(canvas, 0, 0, inputSize, inputSize);
    const data = ctx.getImageData(0, 0, inputSize, inputSize).data;
    const matrix = Array.from({ length: inputSize }, () => Array(inputSize).fill(0));
    for (let y = 0; y < inputSize; y++) {
      for (let x = 0; x < inputSize; x++) {
        const i = (y * inputSize + x) * 4;
        matrix[y][x] = 1 - (data[i] + data[i + 1] + data[i + 2]) / 765;
      }
    }
    return matrix;
  }

  return {
    labels,
    hebrewLabels,
    labelVectors,
    inputSize,
    descriptorSize,
    createRandom,
    makeModel,
    forward,
    trainOne,
    generateShape,
    generateCleanShape,
    generateDataset,
    shuffle,
    canvasToMatrix
  };
})();
