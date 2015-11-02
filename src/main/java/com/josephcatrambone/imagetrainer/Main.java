package com.josephcatrambone.imagetrainer;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.ConvolutionalNetwork;
import com.josephcatrambone.aij.networks.MeanFilterNetwork;
import com.josephcatrambone.aij.networks.RestrictedBoltzmannMachine;
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
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Logger;

/**
 * Created by Jo on 10/31/2015.
 */
public class Main extends Application {
	private static final Logger LOGGER = Logger.getLogger(Main.class.getName());
	public final int WIDTH = 800;
	public final int HEIGHT = 600;

	@Override
	public void start(Stage stage) {
		trainRBM(stage);
	}

	public void trainRBM(Stage stage) {
		final int HIDDEN_SIZE = 400;
		Random random = new Random();

		// Load training data
		Matrix data = loadMNIST("train-images-idx3-ubyte");

		// Build RBM for learning
		final RestrictedBoltzmannMachine rbm = new RestrictedBoltzmannMachine(28 * 28, HIDDEN_SIZE);
		final ContrastiveDivergenceTrainer rbmTrainer = new ContrastiveDivergenceTrainer();
		rbmTrainer.batchSize = 10; // 5x20
		rbmTrainer.learningRate = 0.01;
		rbmTrainer.maxIterations = 10;
		rbmTrainer.gibbsSamples = 1;

		// Try loading the network.
		try (BufferedReader fin = new BufferedReader(new FileReader("rbm.txt"))) {
			Scanner scanner = new Scanner(fin);
			scanner.nextLine();
			String vbString = scanner.nextLine();
			scanner.nextLine();
			String hbString = scanner.nextLine();
			scanner.nextLine();
			String wString = scanner.nextLine();
			rbm.setVisibleBias(NetworkIOTools.stringToMatrix(vbString));
			rbm.setHiddenBias(NetworkIOTools.stringToMatrix(hbString));
			rbm.setWeights(0, NetworkIOTools.stringToMatrix(wString));
			System.out.println("Loaded RBM.");
		} catch (IOException ioe) {
			// Do nothing.
		}

		final Matrix examples = data;

		// Spawn a separate training thread.
		Thread trainerThread = new Thread(() -> {
			int cycles = 0;
			while (true) {
				//synchronized (rbm) {
				rbmTrainer.train(rbm, examples, null, null);
				//}
				// Change training params.
				if (cycles++ > 10000) {
					try (BufferedWriter fout = new BufferedWriter(new FileWriter("rbm.txt"))) {
						fout.write("visible_bias\n");
						fout.write(NetworkIOTools.matrixToString(rbm.getVisibleBias()));
						fout.write("hidden_bias\n");
						fout.write(NetworkIOTools.matrixToString(rbm.getHiddenBias()));
						fout.write("weights\n");
						fout.write(NetworkIOTools.matrixToString(rbm.getWeights(0)));
					} catch (IOException ioe) {

					}
					rbmTrainer.gibbsSamples += 1;
					rbmTrainer.learningRate *= 0.9;
					cycles = 0;
				}

				// Stop if interrupted.
				if (Thread.currentThread().isInterrupted()) {
					return;
				}
				// Yield to give someone else a chance to run.
				try {
					System.out.println(System.currentTimeMillis() + "|" + cycles + "|" + rbmTrainer.gibbsSamples + "|" + rbmTrainer.lastError);
					Thread.sleep(0);
				} catch (InterruptedException ie) {
					// If interrupted, abort.
					return;
				}
			}
		});
		trainerThread.start();

		// Set up UI
		stage.setTitle("Aij Test UI");
		GridPane pane = new GridPane();
		pane.setAlignment(Pos.CENTER);
		Scene scene = new Scene(pane, WIDTH, HEIGHT);
		ImageView imageView = new ImageView(visualizeRBM(rbm, null, true));
		ImageView exampleView = new ImageView(ImageTools.matrixToFXImage(Matrix.random(28, 28), true));
		pane.add(imageView, 0, 0);
		pane.add(exampleView, 1, 0);
		//pane.add(imageView);
		stage.setScene(scene);
		stage.show();

		// Repeated draw.
		Timeline timeline = new Timeline();
		timeline.setCycleCount(Timeline.INDEFINITE);
		timeline.getKeyFrames().add(new KeyFrame(Duration.seconds(0.2), new EventHandler<ActionEvent>() {
			int iteration = 0;
			Matrix input = null;

			@Override
			public void handle(ActionEvent event) {
				if (stage.isFocused() && stage.isShowing()) {
					if (iteration % 100 == 0) {
						iteration = 0;
						// Draw RBM
						//rbmTrainer.train(edgeDetector, data, null, null);
						System.out.println("# Drawing...");
						Image img = visualizeRBM(rbm, null, true);
						imageView.setImage(img);

						// Render a new example
						input = Matrix.random(1, 28 * 28).add_i(1.0).elementMultiply_i(0.5);
					}
					input = rbm.daydream(input, 1);
					Matrix ex = input.clone();
					exampleView.setImage(ImageTools.matrixToFXImage(ex.reshape_i(28, 28), true));
				}
				iteration++;
			}
		}));
		timeline.playFromStart();

		stage.setOnCloseRequest((WindowEvent w) -> {
			timeline.stop();
			trainerThread.interrupt();
		});
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

	public Matrix loadMNIST(final String filename) {
		int numImages = 50000;
		int imgWidth = -1;
		int imgHeight = -1;
		Matrix trainingData = null;

		// Load MNIST training data.
		try(FileInputStream fin = new FileInputStream(filename); DataInputStream din = new DataInputStream(fin)) {
			din.readInt();
			assert(din.readInt() == 2051);
			numImages = Math.min(din.readInt(), numImages);
			imgHeight = din.readInt();
			imgWidth = din.readInt();
			System.out.println("numImages: " + numImages);
			System.out.println("height: " + imgHeight);
			System.out.println("width: " + imgWidth);
			trainingData = new Matrix(numImages, imgWidth*imgHeight);
			for(int i=0; i < numImages; i++) {
				for(int y=0; y < imgHeight; y++) {
					for(int x=0; x < imgWidth; x++) {
						int grey = ((int)din.readByte()) & 0xFF; // Java is always signed.  Need to and with 0xFF to undo it.
						trainingData.set(i, x+(y*imgWidth), ((double)grey)/256.0);
					}
				}
				if(i%1000 == 0) {
					System.out.println("Loaded " + i + " out of " + numImages);
				}
			}
			fin.close();
		} catch(FileNotFoundException fnfe) {
			System.out.println("Unable to find and load file.");
			System.exit(-1);
		} catch(IOException ioe) {
			System.out.println("IO Exception while reading data.");
			System.exit(-1);
		}
		return trainingData;
	}

	public static void main(String[] args) {
		//launch(args);
		trainImageNetwork(args);
	}

	public static void trainImageNetwork(String[] args) {
		final String IMAGE_PATH = "./images";
		final String IMAGE_EXTENSION = ".png";
		final String RBM_0_FILENAME = "rbm0.txt";
		final String RBM_1_FILENAME = "rbm1.txt";
		final int IMAGE_LIMIT = 4000;
		final int IMAGE_WIDTH = 256;
		final int IMAGE_HEIGHT = 256;
		final int STEP = 2;
		final int RBM_SIZE_0 = 4;
		final int RBM_SIZE_1 = 4;
		final int RBM_OUTPUT_0 = 20;
		final int RBM_OUTPUT_1 = 10;

		// Set up our network!
		// One RBM to detect edges and make gabors.  4x4 -> 20x20.  Another 4x4 -> 10x10
		// First conv layer goes from image width/height, samples RBM sized chunks, goes to imwidth/steps * rbm output.
		// Finally, layer4 is a fully connected layer which goes form the (flat) output of layer 3 to an output of two classes.
		RestrictedBoltzmannMachine layer0 = new RestrictedBoltzmannMachine(RBM_SIZE_0*RBM_SIZE_0, RBM_OUTPUT_0*RBM_OUTPUT_0);
		ConvolutionalNetwork layer1 = new ConvolutionalNetwork(layer0, IMAGE_WIDTH, IMAGE_HEIGHT, RBM_SIZE_0, RBM_SIZE_0, RBM_OUTPUT_0, RBM_OUTPUT_0, STEP, STEP, ConvolutionalNetwork.EdgeBehavior.ZEROS);
		RestrictedBoltzmannMachine layer2 = new RestrictedBoltzmannMachine(RBM_SIZE_1*RBM_SIZE_1, RBM_OUTPUT_1*RBM_OUTPUT_1);
		ConvolutionalNetwork layer3 = new ConvolutionalNetwork(layer2, (IMAGE_WIDTH/STEP)*RBM_OUTPUT_0, (IMAGE_HEIGHT/STEP)*RBM_OUTPUT_0, RBM_SIZE_1, RBM_SIZE_1, RBM_OUTPUT_1, RBM_OUTPUT_1, STEP, STEP, ConvolutionalNetwork.EdgeBehavior.ZEROS);
		//NeuralNetwork layer4 = new NeuralNetwork(new int[] {layer3.getNumOutputs(), 100, 2}, new String[] {"tanh", "tanh", "tanh"});

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
		try {
			Files.list(new File(IMAGE_PATH).toPath())
				.filter(p -> p.getFileName().endsWith(IMAGE_EXTENSION) && !p.getFileName().startsWith("."))
				.limit(IMAGE_LIMIT)
				.forEach(p -> examples.appendRow(ImageTools.imageFileToMatrix(p.getFileName().toString(), IMAGE_WIDTH, IMAGE_HEIGHT).reshape_i(1, IMAGE_WIDTH*IMAGE_HEIGHT).getRowArray(0)));
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

		ConvolutionalTrainer convTrainer = new ConvolutionalTrainer();
		convTrainer.operatorTrainer = cdTrainer;
		convTrainer.subwindowsPerExample = 1000;
		convTrainer.examplesPerBatch = 10;
		convTrainer.maxIterations = 1000;

		// Train first layer.
		System.out.println("Examples loaded.  Training layer 0.");
		convTrainer.train(layer1, examples, null, null);

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
		convTrainer.train(layer3, examples2, null, null);

		// Save results.
		try (BufferedWriter fout = new BufferedWriter(new FileWriter(RBM_1_FILENAME))) {
			fout.write("visible_bias\n");
			fout.write(NetworkIOTools.matrixToString(layer2.getVisibleBias()));
			fout.write("hidden_bias\n");
			fout.write(NetworkIOTools.matrixToString(layer2.getHiddenBias()));
			fout.write("weights\n");
			fout.write(NetworkIOTools.matrixToString(layer2.getWeights(0)));
		} catch (IOException ioe) {}

		System.out.println("Done.");
		System.exit(0);
	}
}
