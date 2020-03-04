import {MnistData} from './data.js';
var canvas, ctx, saveButton, clearButton;
var pos = {x:0, y:0};
var rawImage;
var model;
	
function getModel() {
	model = tf.sequential();
	
	// input images are 28x28 pixels in monochrome 
	model.add(tf.layers.conv2d({inputShape: [28, 28, 1], kernelSize: 3, filters: 8, activation: 'relu'}));
	// Pooling layer to reduce dimensionality
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	// Convolution layer with 3x3 size 16 filters to learn patterns, 
	// relu activation is used to filter activations smaller than 0.
	model.add(tf.layers.conv2d({filters: 16, kernelSize: 3, activation: 'relu'}));
	// Pooling layer to reduce dimensionality
	model.add(tf.layers.maxPooling2d({poolSize: [2, 2]}));
	// Flatten 28x28 pixel images to 784 vector for NN input
	model.add(tf.layers.flatten());
	// Fully connected danse layer that learns weights
	model.add(tf.layers.dense({units: 128, activation: 'relu'}));
	// Final output layer with 10 classes and probability distribution between them
	model.add(tf.layers.dense({units: 10, activation: 'softmax'}));


	model.compile(
			// watch for proper syntax for passed Javascript dictionaries {}
			{
			optimizer: tf.train.adam(), 
			loss: 'categoricalCrossentropy', 
			metrics: ['accuracy']
			});
			
	return model;
}

async function train(model, data) {
	// define metrics to follow via callbacks
	const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
	// TFJS VIS library will create DOM object to render the metrics
	const container = { name: 'Model Training', styles: { height: '640px' } };
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  
	const BATCH_SIZE = 512;
	const TRAIN_DATA_SIZE = 5500;
	const TEST_DATA_SIZE = 1000;

	// Once we load the data (from run() function calls) we can batch and rezise them to 28x28
	// To create an array containing training Xs, and Ys
	// tf.tidy() cleans all intermediate tensors after executing function except the one that returns,
	// so d gets cleaned u after is done to save memory
	const [trainXs, trainYs] = tf.tidy(() => {
		// Get the next batch from datasource 5,500
		const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
		return [
			d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
			d.labels //as the labels are allready encodex it will return them as a second elemet
		];
	}); 

	const [testXs, testYs] = tf.tidy(() => {
		const d = data.nextTestBatch(TEST_DATA_SIZE);
		return [
			d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
			d.labels
		];
	});

	return model.fit(trainXs, trainYs, {
		// good idea to use subsets not to block browser with too much data
		batchSize: BATCH_SIZE,
		// Use validationData to see metrics in training process, use logs.val_acc
		validationData: [testXs, testYs],
		epochs: 20,
		// use it to prevent overfitting is similiar classes are in the same batch
		shuffle: true,
		// use it to update user on training process or to visualize training process with tf-vis library
		callbacks: fitCallbacks
	});
}

function setPosition(e){
	pos.x = e.clientX-100;
	pos.y = e.clientY-100;
}
  
// As we draw , when we finish we copy content as raw image  
function draw(e) {
	if(e.buttons!=1) return;
	ctx.beginPath();
	ctx.lineWidth = 24;
	ctx.lineCap = 'round';
	ctx.strokeStyle = 'white';
	ctx.moveTo(pos.x, pos.y);
	setPosition(e);
	ctx.lineTo(pos.x, pos.y);
	ctx.stroke();
	rawImage.src = canvas.toDataURL('image/png');
}
    
function erase() {
	ctx.fillStyle = "black";
	ctx.fillRect(0,0,280,280);
}
    
// When save button is pressed the inference is called	
function save() {
	// We are passing the raw image from draw() saying we want only one channel
	var raw = tf.browser.fromPixels(rawImage,1);
	// Canvas was 280x280px so we want to resize it to fit the model
	var resized = tf.image.resizeBilinear(raw, [28,28]);
	// Resize from 3 dimensional to 4 dimensional to fit the model
	var tensor = resized.expandDims(0);
	// call predict and return class probability distribution
    var prediction = model.predict(tensor);
	// take the highest probability 
    var pIndex = tf.argMax(prediction, 1).dataSync();
    
	alert(pIndex);
}

// Sets up UI    
function init() {
	canvas = document.getElementById('canvas');
	rawImage = document.getElementById('canvasimg');
	ctx = canvas.getContext("2d");
	ctx.fillStyle = "black";
	ctx.fillRect(0,0,280,280);
	canvas.addEventListener("mousemove", draw);
	canvas.addEventListener("mousedown", setPosition);
	canvas.addEventListener("mouseenter", setPosition);
	saveButton = document.getElementById('sb');
	saveButton.addEventListener("click", save);
	clearButton = document.getElementById('cb');
	clearButton.addEventListener("click", erase);
}


async function run() {
    // Initial the data class ad load and transform data from data.js	
	const data = new MnistData();
	await data.load();
	const model = getModel();
	tfvis.show.modelSummary({name: 'Model Architecture'}, model);
	await train(model, data);
	init();
	alert("Training is done, try classifying your handwriting!");
}
// as the document is loaded it will call the run function
document.addEventListener('DOMContentLoaded', run);



    
