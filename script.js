
Math.seedrandom("vfh-ea2");

const N = 100;
const NOISE_STDDEV = Math.sqrt(0.05);
const EPOCHS_BEST = 100;
const EPOCHS_OVER = 600;
const BATCH_SIZE = 32;
const LEARNING_RATE = 0.01;

function func(x) {
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function generateData(withNoise = false) {
    const xVals = Array.from({ length: N }, () => Math.random() * 4 - 2);
    const yVals = xVals.map(func);
    const noise = withNoise ? tf.randomNormal([N], 0, NOISE_STDDEV).arraySync() : Array(N).fill(0);
    const yNoisy = yVals.map((y, i) => y + noise[i]);

    const zipped = xVals.map((x, i) => ({ x, y: yNoisy[i] }));
    tf.util.shuffle(zipped);

    const train = zipped.slice(0, N / 2);
    const test = zipped.slice(N / 2);
    return { train, test };
}

function createModel() {
    const model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1], units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1 })); 
    model.compile({ optimizer: tf.train.adam(LEARNING_RATE), loss: 'meanSquaredError' });
    return model;
}

async function trainModel(model, data, epochs) {
    const xTrain = tf.tensor2d(data.train.map(d => [d.x]));
    const yTrain = tf.tensor2d(data.train.map(d => [d.y]));
    await model.fit(xTrain, yTrain, { epochs, batchSize: BATCH_SIZE, shuffle: true });
    xTrain.dispose();
    yTrain.dispose();
}

async function evaluateModel(model, data) {
    const x = tf.tensor2d(data.map(d => [d.x]));
    const yTrue = tf.tensor2d(data.map(d => [d.y]));
    const yPred = model.predict(x);
    const lossTensor = tf.losses.meanSquaredError(yTrue, yPred);
    const loss = (await lossTensor.data())[0];
    tf.dispose([x, yTrue, yPred, lossTensor]);
    return loss;
}

async function predictY(model, data) {
    return tf.tidy(() => {
        const x = tf.tensor2d(data.map(d => [d.x]));
        const preds = model.predict(x);
        return preds.array();
    });
}

function plotData(divId, title, points1, points2, label1, label2) {
    Plotly.newPlot(divId, [
        {
            x: points1.map(p => p.x),
            y: points1.map(p => p.y),
            mode: 'markers',
            name: label1
        },
        {
            x: points2.map(p => p.x),
            y: points2.map(p => p.y),
            mode: 'markers',
            name: label2
        }
    ], {
        title,
        xaxis: { title: 'x' },
        yaxis: { title: 'y' }
    });
}

function plotPrediction(divId, title, original, predY, label) {
    Plotly.newPlot(divId, [
        {
            x: original.map(p => p.x),
            y: original.map(p => p.y),
            mode: 'markers',
            name: 'Original'
        },
        {
            x: original.map(p => p.x),
            y: predY,
            mode: 'lines',
            name: label
        }
    ], {
        title,
        xaxis: { title: 'x' },
        yaxis: { title: 'y & Prediction' }
    });
}

async function main() {
    const plotsDiv = document.getElementById("plots");
    const dataClean = generateData(false);
    const dataNoisy = generateData(true);

    plotsDiv.innerHTML += '<section id="r1"><h2>R1) Daten</h2><div class="container">' +
        '<div class="plot-container"><div id="dataClean"></div></div>' +
        '<div class="plot-container"><div id="dataNoisy"></div></div></div></section>';
    plotData("dataClean", "Ohne Rauschen", dataClean.train, dataClean.test, "Train", "Test");
    plotData("dataNoisy", "Mit Rauschen", dataNoisy.train, dataNoisy.test, "Train", "Test");

    const modelClean = createModel();
    await trainModel(modelClean, dataClean, 100);
    const lossTrain1 = await evaluateModel(modelClean, dataClean.train);
    const lossTest1 = await evaluateModel(modelClean, dataClean.test);
    const predTrain1 = (await predictY(modelClean, dataClean.train)).map(p => p[0]);
    const predTest1 = (await predictY(modelClean, dataClean.test)).map(p => p[0]);

    plotsDiv.innerHTML += '<section id="r2"><h2>R2) Modell ohne Rauschen</h2><div class="container">' +
        '<div class="plot-container"><div id="modelCleanTrain"></div><div class="loss">Train Loss: ' + lossTrain1.toFixed(4) + '</div></div>' +
        '<div class="plot-container"><div id="modelCleanTest"></div><div class="loss">Test Loss: ' + lossTest1.toFixed(4) + '</div></div></div></section>';
    plotPrediction("modelCleanTrain", "Trainingsdaten", dataClean.train, predTrain1, "Vorhersage");
    plotPrediction("modelCleanTest", "Testdaten", dataClean.test, predTest1, "Vorhersage");

    const modelBest = createModel();
    await trainModel(modelBest, dataNoisy, EPOCHS_BEST);
    const lossTrain2 = await evaluateModel(modelBest, dataNoisy.train);
    const lossTest2 = await evaluateModel(modelBest, dataNoisy.test);
    const predTrain2 = (await predictY(modelBest, dataNoisy.train)).map(p => p[0]);
    const predTest2 = (await predictY(modelBest, dataNoisy.test)).map(p => p[0]);

    plotsDiv.innerHTML += '<section id="r3"><h2>R3) Modell Best-Fit</h2><div class="container">' +
        '<div class="plot-container"><div id="modelBestTrain"></div><div class="loss">Train Loss: ' + lossTrain2.toFixed(4) + '</div></div>' +
        '<div class="plot-container"><div id="modelBestTest"></div><div class="loss">Test Loss: ' + lossTest2.toFixed(4) + '</div></div></div></section>';
    plotPrediction("modelBestTrain", "Trainingsdaten", dataNoisy.train, predTrain2, "Vorhersage");
    plotPrediction("modelBestTest", "Testdaten", dataNoisy.test, predTest2, "Vorhersage");

    const modelOver = createModel();
    await trainModel(modelOver, dataNoisy, EPOCHS_OVER);
    const lossTrain3 = await evaluateModel(modelOver, dataNoisy.train);
    const lossTest3 = await evaluateModel(modelOver, dataNoisy.test);
    const predTrain3 = (await predictY(modelOver, dataNoisy.train)).map(p => p[0]);
    const predTest3 = (await predictY(modelOver, dataNoisy.test)).map(p => p[0]);

    plotsDiv.innerHTML += '<section id="r4"><h2>R4) Modell Overfit</h2><div class="container">' +
        '<div class="plot-container"><div id="modelOverTrain"></div><div class="loss">Train Loss: ' + lossTrain3.toFixed(4) + '</div></div>' +
        '<div class="plot-container"><div id="modelOverTest"></div><div class="loss">Test Loss: ' + lossTest3.toFixed(4) + '</div></div></div></section>';            plotPrediction("modelOverTrain", "Trainingsdaten", dataNoisy.train, predTrain3, "Vorhersage");
    plotPrediction("modelOverTest", "Testdaten", dataNoisy.test, predTest3, "Vorhersage");
}

main();