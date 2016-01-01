package com.josephcatrambone.imagetrainer;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.*;
import com.josephcatrambone.aij.trainers.BackpropTrainer;
import com.josephcatrambone.aij.trainers.ContrastiveDivergenceTrainer;
import com.josephcatrambone.aij.trainers.ConvolutionalTrainer;
import com.josephcatrambone.aij.trainers.DeepNetworkTrainer;
import com.josephcatrambone.aij.utilities.ImageTools;
import com.josephcatrambone.aij.utilities.NetworkIOTools;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.GridPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import javafx.util.Duration;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.logging.Logger;

/**
 * Created by Jo on 10/31/2015.
 */
public class Main extends Application {
	final int IMAGE_WIDTH = 32;
	final int IMAGE_HEIGHT = 32;

	final String IMAGE_PATH = "./images";
	final String POSITIVE_EXAMPLE_PATH = "./positive_examples";
	final String NEGATIVE_EXAMPLE_PATH = "./negative_examples";
	final String IMAGE_EXTENSION = ".png";
	final int IMAGE_LIMIT = 4000;

	private static final Logger LOGGER = Logger.getLogger(Main.class.getName());

	private DeepNetwork buildNetwork() {
		final String FINAL_NETWORK_FILENAME = "dn.dat";
		final int STEP_0 = 1;
		final int STEP_1 = 16;
		final int STEP_2 = 16;
		final int RBM_INPUT = 4;
		final int RBM_OUTPUT = 16; // 16 -> 256

		// If we already trained our network, just load it and return.
		Network trainedNetwork = NetworkIOTools.loadNetworkFromDisk(FINAL_NETWORK_FILENAME);
		if(trainedNetwork != null) {
			return (DeepNetwork)trainedNetwork;
		}

		// Set up our network!
		// One RBM to detect edges and make gabors.  4x4 -> 20x20.  Another 4x4 -> 10x10
		// First conv layer goes from image width/height, samples RBM sized chunks, goes to imwidth/steps * rbm output.
		// Finally, nn is a fully connected layer which goes form the (flat) output of layer 3 to an output of two classes.
		final RestrictedBoltzmannMachine rbm0 = new RestrictedBoltzmannMachine(RBM_INPUT*RBM_INPUT, RBM_OUTPUT*RBM_OUTPUT);
		final ConvolutionalNetwork conv0 = new ConvolutionalNetwork(rbm0,
				IMAGE_WIDTH, IMAGE_HEIGHT,
				RBM_INPUT, RBM_INPUT,
				RBM_OUTPUT, RBM_OUTPUT,
				STEP_0, STEP_0,
				ConvolutionalNetwork.EdgeBehavior.ZEROS);

		final MaxPoolingNetwork mp0 = new MaxPoolingNetwork(RBM_OUTPUT*RBM_OUTPUT*4);
		final ConvolutionalNetwork conv1 = new ConvolutionalNetwork(mp0,
				conv0.getOutputWidth(), conv0.getOutputHeight(),
				RBM_OUTPUT*2, RBM_OUTPUT*2,
				1, 1,
				RBM_OUTPUT, RBM_OUTPUT, // Since each step maxpools two RBM outputs, step sideways this much.
				ConvolutionalNetwork.EdgeBehavior.ZEROS);

		final RestrictedBoltzmannMachine rbm1 = new RestrictedBoltzmannMachine(RBM_INPUT*RBM_INPUT, RBM_OUTPUT*RBM_OUTPUT);
		final ConvolutionalNetwork conv2 = new ConvolutionalNetwork(rbm1,
				conv1.getOutputWidth(), conv1.getOutputHeight(),
				RBM_INPUT,	RBM_INPUT,
				RBM_OUTPUT, RBM_OUTPUT,
				STEP_2, STEP_2,
				ConvolutionalNetwork.EdgeBehavior.ZEROS);

		final MaxPoolingNetwork mp1 = new MaxPoolingNetwork(RBM_OUTPUT*RBM_OUTPUT*4);
		final ConvolutionalNetwork conv3 = new ConvolutionalNetwork(mp1,
				conv2.getOutputWidth(), conv2.getOutputHeight(),
				RBM_OUTPUT*2, RBM_OUTPUT*2,
				1, 1,
				RBM_OUTPUT, RBM_OUTPUT,
				ConvolutionalNetwork.EdgeBehavior.ZEROS);

		final RestrictedBoltzmannMachine rbm2 = new RestrictedBoltzmannMachine(RBM_INPUT*RBM_INPUT, RBM_OUTPUT*RBM_OUTPUT);
		final ConvolutionalNetwork conv4 = new ConvolutionalNetwork(rbm2,
				conv3.getOutputWidth(), conv3.getOutputHeight(),
				RBM_INPUT,	RBM_INPUT,
				RBM_OUTPUT, RBM_OUTPUT,
				STEP_1, STEP_1,
				ConvolutionalNetwork.EdgeBehavior.ZEROS);

		final NeuralNetwork fc = new NeuralNetwork(new int[] {conv4.getNumOutputs(), 100, 2}, new String[] {"tanh", "tanh", "tanh"});

		DeepNetwork dn = new DeepNetwork(conv0, conv1, conv2, conv3, conv4);
		NetworkIOTools.saveNetworkToDisk(dn, FINAL_NETWORK_FILENAME);

		return dn;
	}

	private DeepNetworkTrainer buildTrainer(DeepNetwork net) {
		DeepNetworkTrainer dnt = new DeepNetworkTrainer(net);

		ContrastiveDivergenceTrainer trainer0_0 = new ContrastiveDivergenceTrainer();
		ConvolutionalTrainer trainer0_1 = new ConvolutionalTrainer();
		trainer0_1.operatorTrainer = trainer0_0;
		dnt.setTrainer(0, trainer0_1); // Train layer zero convolution with ct.

		dnt.setTrainer(1, null); // No training for maxpool.

		ContrastiveDivergenceTrainer trainer2_0 = new ContrastiveDivergenceTrainer();
		ConvolutionalTrainer trainer2_1 = new ConvolutionalTrainer();
		trainer0_1.operatorTrainer = trainer2_0;
		dnt.setTrainer(2, trainer2_1); // Train layer zero convolution with ct.

		dnt.setTrainer(3, null); // No training for maxpool.

		ContrastiveDivergenceTrainer trainer4_0 = new ContrastiveDivergenceTrainer();
		ConvolutionalTrainer trainer4_1 = new ConvolutionalTrainer();
		trainer0_1.operatorTrainer = trainer4_0;
		dnt.setTrainer(4, trainer4_1); // Train layer zero convolution with ct.

		return dnt;
	}

	@Override
	public void start(Stage stage) {
		DeepNetwork net = buildNetwork();
		DeepNetworkTrainer dnt = buildTrainer(net);

		// Set up UI
		stage.setTitle("Aij Test UI");
		GridPane pane = new GridPane();
		pane.setAlignment(Pos.CENTER);
		Scene scene = new Scene(pane, IMAGE_WIDTH, IMAGE_HEIGHT);
		//ImageView exampleView = new ImageView(ImageTools.matrixToFXImage(Matrix.random(IMAGE_WIDTH, IMAGE_HEIGHT), true));
		Button makeButton = new Button("Make Image");
		//pane.add(exampleView, 0, 0);
		pane.add(makeButton, 0, 0);
		stage.setScene(scene);
		stage.show();

		// Load our training data.
		System.out.println("Loading images.");
		Matrix examples = Matrix.zeros(1, IMAGE_WIDTH * IMAGE_HEIGHT);
		Matrix positiveExamples = Matrix.zeros(1, IMAGE_WIDTH*IMAGE_HEIGHT);
		Matrix negativeExamples = Matrix.zeros(1, IMAGE_WIDTH*IMAGE_HEIGHT);
		try {
			Files.list(new File(IMAGE_PATH).toPath())
					.filter(p -> p.getFileName().endsWith(IMAGE_EXTENSION) && !p.getFileName().startsWith("."))
					.limit(IMAGE_LIMIT)
					.forEach(p -> examples.appendRow(ImageTools.imageFileToMatrix(p.getFileName().toString(), IMAGE_WIDTH, IMAGE_HEIGHT).reshape_i(1, IMAGE_WIDTH*IMAGE_HEIGHT).getRowArray(0)));
			Files.list(new File(POSITIVE_EXAMPLE_PATH).toPath())
					.filter(p -> p.getFileName().endsWith(IMAGE_EXTENSION) && !p.getFileName().startsWith("."))
					.limit(IMAGE_LIMIT)
					.forEach(p -> positiveExamples.appendRow(ImageTools.imageFileToMatrix(p.getFileName().toString(), IMAGE_WIDTH, IMAGE_HEIGHT).reshape_i(1, IMAGE_WIDTH*IMAGE_HEIGHT).getRowArray(0)));
			Files.list(new File(NEGATIVE_EXAMPLE_PATH).toPath())
					.filter(p -> p.getFileName().endsWith(IMAGE_EXTENSION) && !p.getFileName().startsWith("."))
					.limit(IMAGE_LIMIT)
					.forEach(p -> negativeExamples.appendRow(ImageTools.imageFileToMatrix(p.getFileName().toString(), IMAGE_WIDTH, IMAGE_HEIGHT).reshape_i(1, IMAGE_WIDTH*IMAGE_HEIGHT).getRowArray(0)));
		} catch(IOException ioe) {

		}

		// Now that we have all the input layers trained, let's try and abstract a representation for out positives.

		// Concatenate our examples.  Positive on top.
		/*
		Matrix allAbstractions = Matrix.concatVertically(positiveAbstraction, negativeAbstraction);

		// Make labels by setting the top part of the matrix to [1, 0] and the bottom to [0, 1]
		Matrix labels = new Matrix(allAbstractions.numRows(), 2);
		for(int i=0; i < positiveAbstraction.numRows(); i++) {
			labels.set(i, 0, 1.0);
		}
		for(int i=0; i < negativeAbstraction.numRows(); i++) {
			labels.set(i+positiveAbstraction.numRows(), 1, 1.0);
		}
		 */

		// Spawn a separate training thread.
		Thread trainerThread = new Thread(() -> {

		});

		Thread makerThread = new Thread(() -> {
			String filename = "output_"+ System.nanoTime() +".png";
			Matrix mat = net.reconstruct(Matrix.random(1, net.getNumOutputs()));
			mat.reshape_i(IMAGE_HEIGHT, IMAGE_WIDTH);
			mat = mat.transpose();
			ImageTools.FXImageToDisk(ImageTools.matrixToFXImage(mat, true), filename);
			System.out.println("Wrote file " + filename);
		});

		makeButton.setOnAction((ActionEvent ae) -> makerThread.start());

		// Repeated draw.
		Timeline timeline = new Timeline();
		timeline.setCycleCount(Timeline.INDEFINITE);
		timeline.getKeyFrames().add(new KeyFrame(Duration.seconds(1.0), new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {

			}
		}));
		timeline.playFromStart();

		stage.setOnCloseRequest((WindowEvent w) -> {
			timeline.stop();
			makerThread.interrupt();
		});

		System.out.println("Done.");
	}

	/*** visualizeRBM
	 * Given an RBM as input, return an image which shows the sensitivity of each pathway.
	 * Attempts to produce a square image.
	 * @param rbm
	 * @param normalizeIntensity
	 * @return
	 */
	public Image visualizeRBM(RestrictedBoltzmannMachine rbm, MeanFilterNetwork mean, boolean normalizeIntensity) {
		int outputNeurons = rbm.getNumOutputs();
		int inputNeurons = rbm.getNumInputs();
		int xSampleCount = (int)Math.ceil(Math.sqrt(outputNeurons));
		int subImgWidth = (int)Math.ceil(Math.sqrt(inputNeurons));
		int imgWidth = (int)Math.ceil(Math.sqrt(outputNeurons))*subImgWidth;
		WritableImage output = new WritableImage(imgWidth, imgWidth);
		PixelWriter pw = output.getPixelWriter();

		Matrix weights = rbm.getWeights(0).clone();
		//Matrix vBias = rbm.getVisibleBias().clone();
		//Matrix hBias = rbm.getHiddenBias().clone();

		// Normalize data if needed
		if(normalizeIntensity) {
			for(int j=0; j < weights.numColumns(); j++) {
				Matrix col = weights.getColumn(j);
				col.normalize_i();
				for(int k=0; k < weights.numRows(); k++) {
					weights.set(k, j, col.get(k, 0));
				}
			}
		}

		for(int i=0; i < outputNeurons; i++) {
			int subImgOffsetX = subImgWidth*(i%xSampleCount);
			int subImgOffsetY = subImgWidth*(i/xSampleCount);

			// Rebuild and draw input to image
			for(int j=0; j < weights.numRows(); j++) {
				double val = weights.get(j, i);
				if(val < 0) { val = 0; }
				if(val > 1) { val = 1; }
				pw.setColor(subImgOffsetX + (j%subImgWidth), subImgOffsetY + (j/subImgWidth), Color.gray(val));
			}
		}

		return output;
	}

	public static void main(String[] args) {
		launch(args);
	}
}
