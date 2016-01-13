package com.josephcatrambone.imagetrainer;

import com.josephcatrambone.aij.Matrix;
import com.josephcatrambone.aij.networks.*;
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
	final int IMAGE_WIDTH = 96;
	final int IMAGE_HEIGHT = 96;
	final String FINAL_NETWORK_FILENAME = "dn.dat";
	final String IMAGE_PATH = "D:\\tmp\\black_and_white\\";
	final String IMAGE_EXTENSION = ".png";
	final int IMAGE_LIMIT = 4000;

	private static final Logger LOGGER = Logger.getLogger(Main.class.getName());

	private DeepNetwork buildNetwork() {
		// Set up our network!
		// One RBM to detect edges and make gabors.
		// First conv layer goes from image width/height, samples RBM sized chunks, goes to imwidth/steps * rbm output.
		// Finally, nn is a fully connected layer which goes form the (flat) output of layer 3 to an output of two classes.
		final RestrictedBoltzmannMachine rbm0 = new RestrictedBoltzmannMachine(
				4*4,
				16*16);
		final ConvolutionalNetwork conv0 = new ConvolutionalNetwork(rbm0,
				IMAGE_WIDTH, IMAGE_HEIGHT,
				4, 4, // Filter 4x4
				16, 16, // Out -> 256
				4, 4, // Stride
				ConvolutionalNetwork.EdgeBehavior.ZEROS);

		final MaxPoolingNetwork mp0 = new MaxPoolingNetwork(
				4*4); // Should match filter
		final ConvolutionalNetwork conv1 = new ConvolutionalNetwork(mp0,
				conv0.getOutputWidth(), conv0.getOutputHeight(),
				4, 4, // Filter
				1, 1, // Out (1x1 for mp)
				4, 4, // Stride
				ConvolutionalNetwork.EdgeBehavior.ZEROS);

		final RestrictedBoltzmannMachine rbm1 = new RestrictedBoltzmannMachine(
				8*8,
				16*16);
		final ConvolutionalNetwork conv2 = new ConvolutionalNetwork(rbm1,
				conv1.getOutputWidth(), conv1.getOutputHeight(),
				8, 8,
				16, 16,
				8, 8,
				ConvolutionalNetwork.EdgeBehavior.ZEROS);

		final MaxPoolingNetwork mp1 = new MaxPoolingNetwork(
				32*32);
		final ConvolutionalNetwork conv3 = new ConvolutionalNetwork(mp1,
				conv2.getOutputWidth(), conv2.getOutputHeight(),
				32, 32,
				1, 1,
				16, 16,
				ConvolutionalNetwork.EdgeBehavior.ZEROS);

		final NeuralNetwork fc = new NeuralNetwork(new int[] {conv3.getNumOutputs(), 100, 2}, new String[] {"tanh", "tanh", "tanh"});

		// Just use the first two for now.
		DeepNetwork dn = new DeepNetwork(conv0, conv1, conv2, conv3);

		return dn;
	}

	private DeepNetworkTrainer buildTrainer(DeepNetwork net) {
		DeepNetworkTrainer dnt = new DeepNetworkTrainer(net);

		ContrastiveDivergenceTrainer trainer0_0 = new ContrastiveDivergenceTrainer();
		trainer0_0.batchSize = 10;
		trainer0_0.earlyStopError = -1;
		trainer0_0.learningRate = 0.2;
		trainer0_0.maxIterations = 1001;
		trainer0_0.notificationIncrement = 1000;
		ConvolutionalTrainer trainer0_1 = new ConvolutionalTrainer();
		trainer0_1.operatorTrainer = trainer0_0;
		trainer0_1.subwindowsPerExample = 500;
		trainer0_1.examplesPerBatch = 10;
		trainer0_1.maxIterations = 10000;
		dnt.setTrainer(0, trainer0_1); // Train layer zero convolution with ct.

		/*
		dnt.setTrainer(1, null); // No training for maxpool.

		ContrastiveDivergenceTrainer trainer2_0 = new ContrastiveDivergenceTrainer();
		trainer2_0.batchSize = 10;
		trainer2_0.earlyStopError = -1;
		trainer2_0.learningRate = 0.1;
		trainer2_0.maxIterations = 1001;
		trainer2_0.notificationIncrement = 1000;
		ConvolutionalTrainer trainer2_1 = new ConvolutionalTrainer();
		trainer2_1.operatorTrainer = trainer2_0;
		trainer2_1.earlyStopError = -1;
		trainer2_1.maxIterations = 10000;
		dnt.setTrainer(2, trainer2_1); // Train layer zero convolution with ct.

		dnt.setTrainer(3, null); // No training for maxpool.
		*/
		return dnt;
	}

	@Override
	public void start(Stage stage) {
		// If we already trained our network, just load it and return.
		Network trainedNetwork = NetworkIOTools.loadNetworkFromDisk(FINAL_NETWORK_FILENAME);
		DeepNetwork net;
		if(trainedNetwork != null) {
			net = (DeepNetwork)trainedNetwork;
		} else {
			net = buildNetwork();
		}
		DeepNetworkTrainer dnt = buildTrainer(net);

		// Set up UI
		stage.setTitle("Aij Test UI");
		GridPane pane = new GridPane();
		pane.setAlignment(Pos.CENTER);
		Scene scene = new Scene(pane, IMAGE_WIDTH, IMAGE_HEIGHT);
		//ImageView exampleView = new ImageView(ImageTools.matrixToFXImage(Matrix.random(IMAGE_WIDTH, IMAGE_HEIGHT), true));
		Button makeButton = new Button("Make Image (Hidden -> Visible)");
		Button exportRBMButton = new Button("Save RBMs");
		//pane.add(exampleView, 0, 0);
		pane.add(makeButton, 0, 0);
		pane.add(exportRBMButton, 0, 1);
		stage.setScene(scene);
		stage.show();

		// Load our training data.
		System.out.println("Loading images.");
		Matrix examples = Matrix.zeros(1, IMAGE_WIDTH * IMAGE_HEIGHT);
		try {
			Files.list(new File(IMAGE_PATH).toPath())
					.filter(p -> p.getFileName().toString().toLowerCase().endsWith(IMAGE_EXTENSION))
					//.limit(IMAGE_LIMIT)
					.forEach(p -> {
						String path = p.getFileName().toAbsolutePath().toString().replace("\\", "\\\\"); // For windows.
						Matrix m = ImageTools.imageFileToMatrix(path, IMAGE_WIDTH, IMAGE_HEIGHT);
						if(m != null) {
							examples.appendRow(m.reshape_i(1, IMAGE_WIDTH * IMAGE_HEIGHT).getRowArray(0));
						}
						System.out.println(p.getFileName().toString());
					});
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
			Runnable notification = () -> {
				System.out.println("Trainer notification fired!");
			};
			System.out.println("Training deep network in background.");
			dnt.train(net, examples, null, notification);
			System.out.println("Done training.  Saving network.");
			NetworkIOTools.saveNetworkToDisk(net, FINAL_NETWORK_FILENAME);
			System.out.println("Done with all the things.  Closing thread.");
		});
		trainerThread.start();

		Runnable makerThread = () -> {
			String filename = "output_"+ System.nanoTime() +".png";
			Matrix mat = net.reconstruct(Matrix.random(1, net.getNumOutputs()));
			mat.reshape_i(IMAGE_HEIGHT, IMAGE_WIDTH);
			mat = mat.transpose();
			ImageTools.FXImageToDisk(ImageTools.matrixToFXImage(mat, true), filename);
			System.out.println("Wrote file " + filename);
		};
		makeButton.setOnAction((ActionEvent ae) -> makerThread.run());

		Runnable exportRMB = () -> {
			ConvolutionalNetwork cn = (ConvolutionalNetwork)net.getSubnet(0);
			ImageTools.FXImageToDisk(visualizeRBM((RestrictedBoltzmannMachine)cn.getOperator(), null, true), "rbm0.png");
		};
		exportRBMButton.setOnAction((ActionEvent ae) -> exportRMB.run());

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
			trainerThread.interrupt();
		});

		//try { trainerThread.join(); } catch(InterruptedException ie) {}
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
