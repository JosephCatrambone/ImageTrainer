package com.josephcatrambone.imagetrainer;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.ConvolutionalNetwork;
import com.josephcatrambone.aij.networks.MeanFilterNetwork;
import com.josephcatrambone.aij.networks.NeuralNetwork;
import com.josephcatrambone.aij.networks.RestrictedBoltzmannMachine;
import com.josephcatrambone.aij.trainers.BackpropTrainer;
import com.josephcatrambone.aij.trainers.ContrastiveDivergenceTrainer;
import com.josephcatrambone.aij.trainers.ConvolutionalTrainer;
import com.josephcatrambone.aij.utilities.ImageTools;
import com.josephcatrambone.aij.utilities.NetworkIOTools;
import javafx.animation.KeyFrame;
import javafx.animation.Timeline;
import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.GridPane;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import javafx.util.Duration;

import java.io.*;
import java.nio.file.Files;
import java.util.Scanner;
import java.util.logging.Logger;

/**
 * Created by Jo on 10/31/2015.
 */
public class Main extends Application {
	private static final Logger LOGGER = Logger.getLogger(Main.class.getName());

	@Override
	public void start(Stage stage) {
		final String IMAGE_PATH = "./images";
		final String POSITIVE_EXAMPLE_PATH = "./positive_examples";
		final String NEGATIVE_EXAMPLE_PATH = "./negative_examples";
		final String IMAGE_EXTENSION = ".png";
		final String RBM_0_FILENAME = "rbm0.txt";
		final String RBM_1_FILENAME = "rbm1.txt";
		final int IMAGE_LIMIT = 4000;
		final int IMAGE_WIDTH = 256;
		final int IMAGE_HEIGHT = 256;
		final int STEP_1 = 2;
		final int STEP_3 = 16;
		final int RBM_SIZE_0 = 8;
		final int RBM_OUTPUT_0 = 16; // 64 -> 256
		final int RBM_SIZE_1 = 32;
		final int RBM_OUTPUT_1 = 4; // 1024 -> 16

		// Calculate some layer sizes.
		int LAYER1_OUTPUT_W = RBM_OUTPUT_0 * (IMAGE_WIDTH/STEP_1);
		int LAYER1_OUTPUT_H = RBM_OUTPUT_0 * (IMAGE_HEIGHT/STEP_1);
		int LAYER3_OUTPUT_W = RBM_OUTPUT_1 * (LAYER1_OUTPUT_W/STEP_3);
		int LAYER3_OUTPUT_H = RBM_OUTPUT_1 * (LAYER1_OUTPUT_H/STEP_3);

		// Set up our network!
		// One RBM to detect edges and make gabors.  4x4 -> 20x20.  Another 4x4 -> 10x10
		// First conv layer goes from image width/height, samples RBM sized chunks, goes to imwidth/steps * rbm output.
		// Finally, layer4 is a fully connected layer which goes form the (flat) output of layer 3 to an output of two classes.
		RestrictedBoltzmannMachine layer0 = new RestrictedBoltzmannMachine(RBM_SIZE_0*RBM_SIZE_0, RBM_OUTPUT_0*RBM_OUTPUT_0);
		ConvolutionalNetwork layer1 = new ConvolutionalNetwork(layer0, IMAGE_WIDTH, IMAGE_HEIGHT, RBM_SIZE_0, RBM_SIZE_0, RBM_OUTPUT_0, RBM_OUTPUT_0, STEP_1, STEP_1, ConvolutionalNetwork.EdgeBehavior.ZEROS);
		RestrictedBoltzmannMachine layer2 = new RestrictedBoltzmannMachine(RBM_SIZE_1*RBM_SIZE_1, RBM_OUTPUT_1*RBM_OUTPUT_1);
		ConvolutionalNetwork layer3 = new ConvolutionalNetwork(layer2, LAYER1_OUTPUT_W, LAYER1_OUTPUT_H, RBM_SIZE_1, RBM_SIZE_1, RBM_OUTPUT_1, RBM_OUTPUT_1, STEP_3, STEP_3, ConvolutionalNetwork.EdgeBehavior.ZEROS);
		NeuralNetwork layer4 = new NeuralNetwork(new int[] {LAYER3_OUTPUT_W*LAYER3_OUTPUT_H, 100, 2}, new String[] {"tanh", "tanh", "tanh"});

		// Try to load existing weights if they already exist.
		try (BufferedReader fin = new BufferedReader(new FileReader(RBM_0_FILENAME))) {
			Scanner scanner = new Scanner(fin);
			scanner.nextLine();
			String vbString = scanner.nextLine();
			scanner.nextLine();
			String hbString = scanner.nextLine();
			scanner.nextLine();
			String wString = scanner.nextLine();
			layer0.setVisibleBias(NetworkIOTools.stringToMatrix(vbString));
			layer0.setHiddenBias(NetworkIOTools.stringToMatrix(hbString));
			layer0.setWeights(0, NetworkIOTools.stringToMatrix(wString));
			System.out.println("Loaded layer 0.");
		} catch (IOException ioe) {}

		try (BufferedReader fin = new BufferedReader(new FileReader(RBM_1_FILENAME))) {
			Scanner scanner = new Scanner(fin);
			scanner.nextLine();
			String vbString = scanner.nextLine();
			scanner.nextLine();
			String hbString = scanner.nextLine();
			scanner.nextLine();
			String wString = scanner.nextLine();
			layer2.setVisibleBias(NetworkIOTools.stringToMatrix(vbString));
			layer2.setHiddenBias(NetworkIOTools.stringToMatrix(hbString));
			layer2.setWeights(0, NetworkIOTools.stringToMatrix(wString));
			System.out.println("Loaded layer 2.");
		} catch (IOException ioe) {}

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

		// Configure our trainers.
		// We need a cdTrainer for the RBMs.
		// We need a convTrainer to apply the CD trainer to the RBMs.
		// Make some trainers.
		ContrastiveDivergenceTrainer cdTrainer = new ContrastiveDivergenceTrainer();
		cdTrainer.batchSize = 10;
		cdTrainer.maxIterations = 10;
		cdTrainer.learningRate = 0.1;
		cdTrainer.gibbsSamples = 1;
		cdTrainer.notificationIncrement = 9;
		cdTrainer.regularization = 0.0001;

		ConvolutionalTrainer convTrainer = new ConvolutionalTrainer();
		convTrainer.operatorTrainer = cdTrainer;
		convTrainer.subwindowsPerExample = 100;
		convTrainer.examplesPerBatch = 100;
		convTrainer.maxIterations = 100;

		// Train first layer.
		System.out.println("Examples loaded.  Training layer 0.");
		convTrainer.train(layer1, examples, null, () -> {
			System.out.println("Contrastive Divergence Training Error: " + cdTrainer.lastError);
			System.out.println("RBM Energy: " + layer0.getFreeEnergy());
		});

		// Save first layer.
		//NetworkIOTools.saveNetworkToDisk(layer0, "layer0.model");
		try (BufferedWriter fout = new BufferedWriter(new FileWriter(RBM_0_FILENAME))) {
			fout.write("visible_bias\n");
			fout.write(NetworkIOTools.matrixToString(layer0.getVisibleBias()));
			fout.write("hidden_bias\n");
			fout.write(NetworkIOTools.matrixToString(layer0.getHiddenBias()));
			fout.write("weights\n");
			fout.write(NetworkIOTools.matrixToString(layer0.getWeights(0)));
		} catch (IOException ioe) {}

		// Abstract the data by one level.
		Matrix examples2 = layer1.predict(examples);

		// Train second layer.
		convTrainer.train(layer3, examples2, null, () -> {
			System.out.println("Contrastive Divergence Training Error: " + cdTrainer.lastError);
			System.out.println("RBM Energy: " + layer2.getFreeEnergy());
		});

		// Save results.
		try (BufferedWriter fout = new BufferedWriter(new FileWriter(RBM_1_FILENAME))) {
			fout.write("visible_bias\n");
			fout.write(NetworkIOTools.matrixToString(layer2.getVisibleBias()));
			fout.write("hidden_bias\n");
			fout.write(NetworkIOTools.matrixToString(layer2.getHiddenBias()));
			fout.write("weights\n");
			fout.write(NetworkIOTools.matrixToString(layer2.getWeights(0)));
		} catch (IOException ioe) {}

		// Now that we have all the input layers trained, let's try and abstract a representation for out positives.
		Matrix temp = layer1.predict(positiveExamples);
		Matrix positiveAbstraction = layer3.predict(temp);
		temp = layer1.predict(negativeExamples);
		Matrix negativeAbstraction = layer3.predict(temp);

		// Concatenate our examples.  Positive on top.
		Matrix allAbstractions = new Matrix(
				positiveAbstraction.numRows() + negativeAbstraction.numRows(),
				positiveAbstraction.numColumns()); // pos/negative have same number of columns.
		allAbstractions.setSubmatrix_i(positiveAbstraction, 0, 0);
		allAbstractions.setSubmatrix_i(negativeAbstraction, positiveAbstraction.numRows(), 0);

		// Make labels by setting the top part of the matrix to [1, 0] and the bottom to [0, 1]
		Matrix labels = new Matrix(allAbstractions.numRows(), 2);
		for(int i=0; i < positiveAbstraction.numRows(); i++) {
			labels.set(i, 0, 1.0);
		}
		for(int i=0; i < negativeAbstraction.numRows(); i++) {
			labels.set(i+positiveAbstraction.numRows(), 1, 1.0);
		}

		// TODO: We need a matrix concat function and a shuffle.

		// Train.
		BackpropTrainer nnTrainer = new BackpropTrainer();
		nnTrainer.batchSize = 10;
		nnTrainer.learningRate = 0.1;
		nnTrainer.maxIterations = 10000;
		nnTrainer.notificationIncrement = 100;

		nnTrainer.train(layer4, allAbstractions, labels, () -> {
			System.out.println("NN Trainer Last Error: " + nnTrainer.lastError);
		});

		// Spawn a separate training thread.
		Thread trainerThread = new Thread(() -> {});
		//trainerThread.start();

		// Set up UI
		stage.setTitle("Aij Test UI");
		GridPane pane = new GridPane();
		pane.setAlignment(Pos.CENTER);
		Scene scene = new Scene(pane, IMAGE_WIDTH, IMAGE_HEIGHT);
		ImageView exampleView = new ImageView(ImageTools.matrixToFXImage(Matrix.random(28, 28), true));
		pane.add(exampleView, 0, 0);
		stage.setScene(scene);
		stage.show();

		// Repeated draw.
		Timeline timeline = new Timeline();
		timeline.setCycleCount(Timeline.INDEFINITE);
		timeline.getKeyFrames().add(new KeyFrame(Duration.seconds(0.2), new EventHandler<ActionEvent>() {
			@Override
			public void handle(ActionEvent event) {
			}
		}));
		timeline.playFromStart();

		stage.setOnCloseRequest((WindowEvent w) -> {
			timeline.stop();
			trainerThread.interrupt();
		});

		System.out.println("Done.");
		System.exit(0);
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
