package com.numlern.app;
import java.io.*;
import java.util.*;
import java.io.*;
import com.opencsv.CSVWriter; 
import Jama.*;

public class NN {
	private double[][] train_x;
	private int[] train_y_raw;
	//private double[] train_y;
	private double[][] train_y;
	private int[] sizes;
	private Layer[] layers;
	private double[][] plot1;
	private double[][] plot2;
	private int plotindex;

	public NN (int[] sizes){
		// The ith element in sizes represents the
		// number of neurons in the ith layer
		// (the first layer is an input layer)
		this.sizes = sizes;

		// don't include input layer
		this.layers = new Layer[this.sizes.length-1];

		// construct all the layers
		for(int i = 0; i < this.layers.length; i++){
			// First argument is the number of neurons in that layer
			// Second argument is the number of inputs for each neuron
			this.layers[i] = new Layer(this.sizes[i+1], this.sizes[i]);
		}

		// Read CSV 
		// Fill network with weights and biases
		
	}


	public int predict(double[] input){
		// input is a 28x28 array

		double[] a = input.clone();

		// for each layer
		for (int i = 0; i < this.layers.length; i++){

			double[] aprime = new double[layers[i].neurons.length];

			// compute the ouput of each neuron in that layer
			for(int j = 0; j < layers[i].neurons.length; j++)
				aprime[j] = activate(layers[i].neurons[j].w, a, layers[i].neurons[j].b, true);
			
			// move to the next layer with
			// the new activation values (outputs)
			a = aprime.clone();

		}

		double max = a[0];
		int res = 0;
		System.out.println(Arrays.toString(a));
		for (int i = 1; i < a.length; i++){
			if (a[i] > max) {
				max = a[i];
				res = i;
			}
		}
		return res;
	}




 	public double activate(double[] w, double[] a, double bias, boolean sig){
 		// ACTIVATE NEURON
 		// compute dot product W * A with bias

 		if (w.length != a.length) throw new RuntimeException("activate(): Vector lengths do not match.");

 		double res = 0;
 		for(int i = 0; i < w.length; i++){
 			res += w[i]*a[i];
 		}
 		res += bias;
 		// optionally return sigmoid
 		if(sig) return sigmoid(res);
 		else return res;
 	}


 	public void train(int epochs, double eta, int minibatch_size, int train_size){
 		this.train_x = new double[train_size][784];
 		this.train_y_raw = new int[train_size];
 		this.train_y = new double[train_size][];
 		this.plotindex = 0;
 		this.plot1 = new double[epochs*train_size][];
 		this.plot2 = new double[epochs*train_size][];
 		//this.train_y = new double[2];

 		System.out.println("Training...");

 		readTrainingData();
 		randomizeParams();

 		for(int j = 0; j < epochs; j++){

 			int num_minibatches = this.train_x.length/minibatch_size;

 			Tuple[] minibatches = generateMiniBatches(minibatch_size);

	 		for(int i = 0; i < num_minibatches; i++){
	 			updateMiniBatch(minibatches[i], eta);
	 		}
	 		System.out.println("-------------------------------");
	 		System.out.println("EPOCH " + (j+1) + "/" + epochs + " COMPLETE");
	 		System.out.println("-------------------------------");
	 		//test();
 		}
 		writeFile("plotL", this.plot1);
 		writeFile("plotL-1", this.plot2);
	 	writeWeights();
	 	writeBiases();
 	}


 	public Tuple[] generateMiniBatches(int minibatch_size){
 		// Create randomized mini batches
 		shuffle(this.train_x, this.train_y);
 		int num_minibatches = this.train_x.length/minibatch_size;
		// Tuple class was reused to store minibatches
		// (slightly better than making a new MiniBatch class... don't worry about it)
		Tuple[] minibatches = new Tuple[num_minibatches];

		for(int i = 0; i < num_minibatches; i++){
 			double[][] xtemp = new double[minibatch_size][this.train_x[0].length];
 			double[][] ytemp = new double[minibatch_size][this.train_y[0].length];

 			for(int k = 0; k < minibatch_size; k++){
 				xtemp[k] = this.train_x[k+i*minibatch_size];
 				ytemp[k] = this.train_y[k+i*minibatch_size];
 			}

 			minibatches[i] = new Tuple(xtemp, ytemp);
 			
 		}
 		return minibatches;
 	}


 	public void updateMiniBatch(Tuple minibatch, double eta){
 		// Create a 2D jagged sum-of-grad(b) matrix
 		double[][] sum_b = new double[this.layers.length][];

 		for(int i = 0; i < this.layers.length; i++)
 			sum_b[i] = new double[this.layers[i].neurons.length];

 		// Create a 3D jagged sum-of-grad(w) matrix
 		double[][][] sum_w = new double[this.layers.length][][];

 		for(int i = 0; i < this.layers.length; i++)
 			sum_w[i] = new double[this.layers[i].neurons.length][];

 		for(int i = 0; i < this.layers.length; i++){
 			for(int j = 0; j < this.layers[i].neurons.length; j++){
 				sum_w[i][j] = new double[this.layers[i].neurons[j].w.length];
 			}
 		}

 		// Zero everything
 		zero(sum_w);
 		zero(sum_b);

 		// For each minibatch,
 		// back-propagate to obtain gradients and sum them
 		for(int i = 0; i < minibatch.x.length; i++){

 			Tuple grad = backProp(minibatch.x[i], minibatch.y[i]);

 			// biases:
 			for(int l = 0; l < sum_b.length; l++){
 				for(int j = 0; j < sum_b[l].length; j++){
 					sum_b[l][j] += grad.b[l][j];
 				}
 			}

 			// weights:
 			for(int l = 0; l < sum_w.length; l++){
 				for(int j = 0; j < sum_w[l].length; j++){
 					for(int k = 0; k < sum_w[l][j].length; k++){
 						sum_w[l][j][k] += grad.w[l][j][k];
 					}
 				}
 			}
 		}

 		// Update weights and biases
 		double mblength = (double) minibatch.x.length;
 		// weights:
 		for(int l = 0; l < this.layers.length; l++){
 			for(int j = 0; j < this.layers[l].neurons.length; j++){
 				for(int k = 0; k < this.layers[l].neurons[j].w.length; k++){

 					//double w = this.layers[l].neurons[j].w[k];
 					
 					//System.out.println(sum_w[l][j][k]);
 					this.layers[l].neurons[j].w[k] -= (eta/mblength)*sum_w[l][j][k];

 				}
 			}
 		}
 		//System.out.println("WEIGHT: " + this.layers[0].neurons[20].w[0]);
 		//System.out.println("WEIGHT: " + this.layers[1].neurons[3].w[18]);
 		//System.out.println("WEIGHT: " + this.layers[0].neurons[12].w[4]);
 		// biases:
 		for(int l = 0; l < this.layers.length; l++){
 			for(int j = 0; j < this.layers[l].neurons.length; j++){
 				//double b = this.layers[l].neurons[j].b;
 				this.layers[l].neurons[j].b -= (eta/mblength)*sum_b[l][j];
 			}
 		}
 	}

 	public Tuple backProp(double[] x, double[] y){
 		// Create a 2D jagged grad(b) matrix
 		double[][] grad_b = new double[this.layers.length][];

 		for(int i = 0; i < this.layers.length; i++)
 			grad_b[i] = new double[this.layers[i].neurons.length];

 		// Create a 3D jagged grad(w) matrix
 		double[][][] grad_w = new double[this.layers.length][][];

 		for(int i = 0; i < this.layers.length; i++)
 			grad_w[i] = new double[this.layers[i].neurons.length][];

 		for(int i = 0; i < this.layers.length; i++){
 			for(int j = 0; j < this.layers[i].neurons.length; j++){
 				grad_w[i][j] = new double[this.layers[i].neurons[j].w.length];
 			}
 		}

 		// Zero everything
 		zero(grad_w);
 		zero(grad_b);

 		// Collect all the activations during a forward pass through the net
 		// ('a' is in order of layers, i.e. a[0] -> activations in layer[0])

 		double[][] a = new double[this.layers.length][];
 		double[][] z = new double[this.layers.length][];
 		forwardPass(x, a, z); // This fills 'a' and 'z'

 		// Now we apply the four fundamental backward pass equations

 		// 1. Get delta error in output (final) layer
 		int output_length = this.layers[this.layers.length-1].neurons.length;

 		double[][] delta = new double[this.layers.length][];
 		for(int i = 0; i < delta.length; i++)
 			delta[i] = new double[this.layers[i].neurons.length];
 		int L = grad_b.length-1;
 		delta[L] = hadamard( costPrime(a[a.length-1],y), sigmoidPrime(z[L]) );
 		
 		// ***
		// for(int j = 0; j < delta[L].length; j++){
		// 	if (delta[L][j] < 0)
		// 		delta[L][j] = -1*Math.log(Math.abs(delta[L][j]));
		// 	else delta[L][j] = Math.log(delta[L][j]);
		// }
 		// double[] grad_approx_b = new double[delta[L].length];
 		// for(int i = 0l < grad_approx_b.length; i++)
 		// 	grad_approx_b[i] = gradApprox(a[a.length-1], )

 		//System.out.println("APPROX:");
 		//System.out.println(Arrays.toString(grad_approx_b));
 		//System.out.println("DELTA:");
 		//System.out.println(Arrays.toString(delta[L]));

 		grad_b[L] = delta[L];
		grad_w[L] = dot(delta[L], a[L-1]);
		

 		
 		// Back propagate through the network to fill grads
 		for(int l = L-1; l >= 0; l--){
 			double [] sig = sigmoidPrime(z[l]);
 			double [][] wT = transpose(this.layers[l+1].weights());
 			double [] prod = dot(wT,delta[l+1]);
 			delta[l] = hadamard(prod, sig);
			// for(int j = 0; j < delta[l].length; j++){
			// 	if (delta[l][j] < 0)
			// 		delta[l][j] = -1*Math.log(Math.abs(delta[l][j]));
			// 	else delta[l][j] = Math.log(delta[l][j]);
			// }
 			// Update gradients
 			//System.out.println(Arrays.toString(delta[l]));
 			//System.out.println(Arrays.toString(delta[l]));
 			grad_b[l] = delta[l];
 			if (l-1 < 0) grad_w[l] = dot(delta[l], x);
 			else grad_w[l] = dot(delta[l], a[l-1]);
 		}
 		
 		this.plot2[plotindex] = delta[L-1];
 		this.plot1[plotindex] = delta[L];

 		// Package grad_b, grad_w in a tuple and return it
 		Tuple grad = new Tuple(grad_b, grad_w);
 		plotindex++;
 		return grad;

 		// END OF BACKPROP
 		
 	}

	public void forwardPass(double[] input, double[][] activations, double[][] z){
		// input is a 28x28 array
		double[] a = input.clone();

		// for each layer
		for (int i = 0; i < this.layers.length; i++){

			double[] atemp = new double[this.layers[i].neurons.length];
			z[i] = new double[this.layers[i].neurons.length];
			// compute the ouput of each neuron in that layer
			for(int j = 0; j < this.layers[i].neurons.length; j++){
				// save the z values (weighted + biased inputs)
				z[i][j] = activate(this.layers[i].neurons[j].w, a, this.layers[i].neurons[j].b, false);
				atemp[j] = sigmoid(z[i][j]);
			}
			// move to the next layer with
			// the new activation values (outputs)
			a = atemp;
			// save the new activation values
			activations[i] = a.clone();
		}
		// System.out.println("A: " + activations[0][10]);
		// System.out.println("A: " + activations[0][20]);
	}


	// // // // // // //
	// MATH FUNCTIONS //
    // // // // // // //

 	public void randomizeParams(){
 		for(int i = 0; i < this.layers.length; i++){
 			for(int j = 0; j < this.layers[i].neurons.length; j++){
 				// randomize bias
 				this.layers[i].neurons[j].b = randomGauss();

 				for(int k = 0; k < this.layers[i].neurons[j].w.length; k++){
 					this.layers[i].neurons[j].w[k] = randomGauss();
 				}
 			}
 		}
 	}

 	public double randomGauss(){
 		Random rand = new Random();
 		return rand.nextGaussian();
 		//return 0.0;
 	}

 	public double gradientApprox(double v, double y){
 		double epsilon = 0.0005;
 		double dC = cost(v + epsilon, y) - cost(v + epsilon, y);
 		double dv = 2*epsilon;
 		return dC/dv;
 	}

 	// Mean squared error (only used for debugging)
 	public double cost(double a, double y){
 		// error = new double[a.length];
 		//for(int i = 0; i < error.length; i++)
 			//error[i] = (a[i] - y[i])*(a[i] - y[i]);
 		//return error;
 		return ((a - y)*(a - y))/2;
 	}

 	// Derivative of mean squared error
 	public double[] costPrime(double[] a, double[] y){
 		// Compare each output activations 'a' with the desired y
 		if (a.length != y.length) throw new RuntimeException("costPrime(): Vector lengths do not match.");
 		double[] error = new double[a.length];
 		for(int i = 0; i < error.length; i++)
 			error[i] = a[i] - y[i];
 		return error;
 	}

 	public double sigmoid(double z){
 		return 1.0/(1.0+Math.exp(-z));
 	}

 	public double[] sigmoidPrime(double[] z){
 		// Derivative of sigmoid function

 		// Apply sigmoid prime to each element
 		double[] res = new double[z.length];
 		for(int i = 0; i < z.length; i++)
 			res[i] = sigmoid(z[i])*(1-sigmoid(z[i]));
 		return res;
 	}

 	public double[] hadamard(double[] A, double [] B){
 		// compute Hadamard product (element-wise multiplication)
 		if (A.length != B.length) throw new RuntimeException("hadamard(): Vector lengths do not match.");

 		double[] res = new double[A.length];
 		for(int i = 0; i < A.length; i++)
 			res[i] = A[i]*B[i];
 		return res;
 	}

 	public double[][] transpose(double[][] arr){
 		double[][] res = new double[arr[0].length][arr.length];
 		Matrix m = new Matrix(arr);
 		res = m.transpose().getArray();
 		return res;
 	}

 	public double[] dot(double[][] A, double[] B){

 		if (A[0].length != B.length) throw new RuntimeException("dot(): Vector lengths do not match."); 		

 		double[] res = new double[A.length];

 		for(int i = 0; i < A.length; i++){
 			double sum = 0;
 			for(int j = 0; j < A[i].length; j++){
 				sum += A[i][j]*B[j];
 			}
 			res[i] = sum;
 		}

 		return res;
 	}

 	public double[][] dot(double[] A, double[] B){
 		// A is a column vector and B is a row vector
 		double[][] res = new double[A.length][B.length];

 		for(int i = 0; i < A.length; i++){
 			for(int j = 0; j < B.length; j++){
 				res[i][j] = A[i]*B[j];
 			}
 		}
 		return res;
 	}

	// Implementing Fisher-Yates Modern Shuffle
	public void shuffle(double[][] x, double[][] y){
		for(int i = 0; i < x.length; i++){
			int switcheroo = i + (int) Math.random()*(x.length - i + 1);

			// Swap two rows of x randomly
			double[] tempx = x[i];
			x[i] = x[switcheroo];
			x[switcheroo] = tempx;

			// Swap two elements of y in exactly the same way
			double[] tempy = y[i];
			y[i] = y[switcheroo];
			y[switcheroo] = tempy;			
		}
	}

	public void zero(double[][] arr){
		for(int i = 0; i < arr.length; i++){
			for(int j = 0; j < arr[i].length; j++)
				arr[i][j] = 0;
		}
	}

	public void zero(double[][][] arr){
		for(int i = 0; i < arr.length; i++){
			for(int j = 0; j < arr[i].length; j++){
				for(int k = 0; k < arr[i][j].length; k++)
					arr[i][j][k] = 0;
			}
		}
	}

	public double[] normalize(int[] arr, double xmin, double xmax, double a, double b){
		// normalize over [a,b]
		double[] res = new double[arr.length];
		for(int i = 0; i < arr.length; i++){
			res[i] = (b-a)*((arr[i] - xmin)/(xmax - xmin)) + a;
		}
		return res;
	}

	public double[][] normalize(double[][] arr, double xmin, double xmax, double a, double b){
		// normalize over [a,b]
		double[][] res = new double[arr.length][arr[0].length];
		for(int i = 0; i < arr.length; i++){
			for(int j = 0; j < arr[i].length; j++)
				res[i][j] = (b-a)*((arr[i][j] - xmin)/(xmax - xmin)) + a;
		}
		return res;
	}

	public double[] vectorize(int input, int range){
		double[] res = new double[range];
		for(int i = 0; i < range; i++)
			res[i] = 0;	
		res[input] = 1;
		return res;
	}

	// // // // // // //
	//  IO FUNCTIONS  //
    // // // // // // //

	public void readTrainingData(){
		// Retrieve training data from mnist_train.CSV
		try{
			Scanner scanIn = new Scanner(new BufferedReader(new InputStreamReader(this.getClass().getResourceAsStream("/mnist_train.csv"))));

			int i = 0;
			while(scanIn.hasNextLine()){
				String inputLine = scanIn.nextLine();

				String[] inArray = inputLine.split(",");

				for(int j = 0; j < inArray.length; j++){
					if(j == 0){
						this.train_y_raw[i] = Integer.parseInt(inArray[j]);
					}
					else{
						this.train_x[i][j - 1] = Double.parseDouble(inArray[j]);
					}
				}

				i++;
			}
		}catch (Exception e){
			System.out.println(e);
		}


		//this.train_y = normalize(this.train_y_raw, 0, 9, 0, 1);
		for(int i = 0; i < this.train_y_raw.length; i++)
			this.train_y[i] = vectorize(this.train_y_raw[i], 10);
		// this.train_x = normalize(this.train_x, 0, 255, -1, 1);

	}

	public void readWeights(){
		// Retrieve pre-trained weight data
		for(int l = 0; l < this.layers.length; l++){
			String filename = "layer_" + l + "_w";
			try{
				Scanner scanIn = new Scanner(new BufferedReader(new InputStreamReader(this.getClass().getResourceAsStream("/mnist_train.csv"))));

				int j = 0;
				while(scanIn.hasNextLine()){
					String inputLine = scanIn.nextLine();

					String[] inArray = inputLine.split(",");

					for(int k = 0; k < inArray.length; k++)
						this.layers[l].neurons[j].w[k] = Double.parseDouble(inArray[k]);

					j++;
				}
			}catch (Exception e){
				System.out.println(e);
			}
		}
	}

	public void readBiases(){
		// Retrieve pre-trained weight data
		for(int l = 0; l < this.layers.length; l++){
			String filename = "layer_" + l + "_w";
			try{
				Scanner scanIn = new Scanner(new BufferedReader(new InputStreamReader(this.getClass().getResourceAsStream("/mnist_train.csv"))));

				int i = 0;
				while(scanIn.hasNextLine()){
					String inputLine = scanIn.nextLine();

					String[] inArray = inputLine.split(",");

					for(int j = 0; j < inArray.length; j++)
						this.layers[l].neurons[j].b = Double.parseDouble(inArray[j]);

					i++;
				}
			}catch (Exception e){
				System.out.println(e);
			}
		}
	}


	public void writeWeights(){
		for(int l = 0; l < this.layers.length; l++){		
			File file = new File("layer_" + l + "_w"); 
		    try { 
		        FileWriter outputfile = new FileWriter(file); 
		        CSVWriter writer = new CSVWriter(outputfile, ',', CSVWriter.NO_QUOTE_CHARACTER); 
		  
		        for(int i = 0; i < this.layers[l].weights().length; i++){

		        	String[] str = new String[this.layers[l].weights()[0].length];

		        	for(int k = 0; k < this.layers[l].weights()[i].length; k++)
		        		str[k] = Double.toString(this.layers[l].weights()[i][k]);

		        	writer.writeNext(str); 
		        }
		  
		        writer.close(); 
		    } 
		    catch (IOException e) { 
		        e.printStackTrace(); 
		    } 
		}
	}

	public void writeBiases(){
		for(int l = 0; l < this.layers.length; l++){		
			File file = new File("layer_" + l + "_b"); 
		    try { 
		        FileWriter outputfile = new FileWriter(file); 
		        CSVWriter writer = new CSVWriter(outputfile, ',', CSVWriter.NO_QUOTE_CHARACTER); 
		  
		  		String[] str = new String[this.layers[l].biases().length];
		  		for(int i = 0; i < this.layers[l].biases().length; i++)
		  			str[i] = Double.toString(this.layers[l].biases()[i]);

		        writer.writeNext(str); 
		  
		        writer.close(); 
		    } 
		    catch (IOException e) { 
		        e.printStackTrace(); 
		    } 
		}
	}

	public void writeFile(String filename, double[][] data){
		File file = new File(filename); 
	    try { 
	        FileWriter outputfile = new FileWriter(file); 
	        CSVWriter writer = new CSVWriter(outputfile, ',', CSVWriter.NO_QUOTE_CHARACTER); 
	  
	  		for(int j = 0; j < data.length; j++){
		  		String[] str = new String[data[j].length];

		  		for(int i = 0; i < data[j].length; i++)
		  			str[i] = Double.toString(data[j][i]);

		        writer.writeNext(str); 
	  		}
	        writer.close(); 
	    } 
	    catch (IOException e) { 
	        e.printStackTrace(); 
	    } 
	}

	public void writeFile(String filename, double[] data){
		File file = new File(filename); 
	    try { 
	        FileWriter outputfile = new FileWriter(file); 
	        CSVWriter writer = new CSVWriter(outputfile, ',', CSVWriter.NO_QUOTE_CHARACTER); 
	  
	  		String[] str = new String[data.length];
	  		for(int i = 0; i < data.length; i++)
	  			str[i] = Double.toString(data[i]);

	        writer.writeNext(str); 
	  
	        writer.close(); 
	    } 
	    catch (IOException e) { 
	        e.printStackTrace(); 
	    } 
	}

	public void test(int test_size){
		double[][] test_x = new double[test_size][784];
		int[] test_y = new int[test_size];
		try{
			Scanner scanIn = new Scanner(new BufferedReader(new InputStreamReader(this.getClass().getResourceAsStream("/mnist_test.csv"))));

			int i = 0;
			while(scanIn.hasNextLine()){
				String inputLine = scanIn.nextLine();

				String[] inArray = inputLine.split(",");

				for(int j = 0; j < inArray.length; j++){
					if(j == 0){
						test_y[i] = Integer.parseInt(inArray[j]);
					}
					else{
						test_x[i][j - 1] = Double.parseDouble(inArray[j]);
					}
				}

				i++;
			}
		}catch (Exception e){
			System.out.println(e);
		}

		// test_x = normalize(test_x, 0, 255, -1, 1);

		int count = 0;
		for(int i = 0; i < test_y.length; i++){
			System.out.println("Prediction : " + this.predict(test_x[i]) + " : in fact : " + test_y[i]);
			if(this.predict(test_x[i]) == test_y[i]){
				count++;
			}
		}

		System.out.println(((double) count) / test_y.length);
	}


}