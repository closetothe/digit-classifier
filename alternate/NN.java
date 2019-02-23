package com.numlern.app;
import java.io.*;
import java.util.*;
import java.io.*;
import com.opencsv.CSVWriter; 
import Jama.*;


/*
	Alternate (more standard) implementation
	using only matrices, and no Layer class.
	[SIMPLEST POSSIBLE]
*/
public class NN {
	private Matrix[] weights;
	private Matrix[] biases;
	private List<Tuple> train;
	private List<Tuple> test;
	private int layers;

	public NN (int[] sizes){
		this.layers = sizes.length;
		// Each layer except the first has weights and biases
		this.weights = new Matrix[layers-1];
		this.biases = new Matrix[layers-1];
		for(int i = 0; i < this.layers-1; i++){
			Random rand = new Random();
			this.weights[i] = Matrix.random(sizes[i+1],sizes[i]);
			this.biases[i] = Matrix.random(sizes[i+1], 1);
			//this.weights[i] = this.weights[i].timesEquals(rand.nextGaussian());
			//this.biases[i] = this.biases[i].timesEquals(rand.nextGaussian());
		}
	}

	public int predict(double[] x){
		// x is a 784-pixel array
		// Convert to column vector
		Matrix a = new Matrix(x.length, 1);
		for(int i = 0; i < x.length; i++)
			a.set(i, 0, x[i]);		
		// Pass through the network
		for(int l = 0; l < layers-1; l++){
			a = sigmoid(this.weights[l].times(a).plus(this.biases[l]));
		}
		// arg max of final a matrix
		double max = a.get(0,0);
		int res = 0;
		for(int i = 0; i < 10; i++){
			if (a.get(i,0) > max){
				max = a.get(i,0);
				res = i;
			}
		}
		return res;
	}

	public void test(){
		this.test = new ArrayList<Tuple>();
		readDataset("mnist_test", this.test);

		int count = 0;
		for(int i = 0; i < this.test.size(); i++){
			int prediction = predict(this.test.get(i).x);
			System.out.println("Prediction : " + prediction + " : in fact : " + argmax(this.test.get(i).y));
			if(prediction == argmax(this.test.get(i).y))
				count++;
		}
		System.out.println(((double) count) / this.test.size());
	}


	// mbsize = minibatch size
	public void train(int epochs, double eta, int mbsize){
		System.out.println("Reading dataset...");
		this.train = new ArrayList<Tuple>();
		readDataset("mnist_train", this.train);
			
		System.out.println("Training...");
		for(int j = 0; j < epochs; j++){
			Collections.shuffle(this.train);
			// A minibatch is just a portion of the shuffled train list
			for(int i = 0; i < this.train.size() - mbsize; i+= mbsize)
				update(this.train.subList(i, i+mbsize), eta);

	 		System.out.println("-------------------------------");
	 		System.out.println("EPOCH " + (j+1) + "/" + epochs + " COMPLETE");
	 		System.out.println("-------------------------------");
		}
	}

	public void update(List<Tuple> minibatch, double eta){
		Matrix[] sum_b = new Matrix[layers-1];
		Matrix[] sum_w = new Matrix[layers-1];
		for(int l = 0; l < layers-1; l++){
			sum_b[l] = new Matrix(this.biases[l].getRowDimension(), 1);
			sum_w[l] = new Matrix(this.weights[l].getRowDimension(), 
									this.weights[l].getColumnDimension());
		}
		for(int i = 0; i < minibatch.size(); i++){
			// Compute gradients
			//System.out.println(Arrays.toString(minibatch.get(i).y));
			Tuple delta = backProp(minibatch.get(i)); 
			// Sum gradients
			for(int l = 0; l < layers-1; l++){
				sum_b[l] = sum_b[l].plus(delta.b[l]);
				sum_w[l] = sum_w[l].plus(delta.w[l]);
			}
		}
		// Update weights and biases
		double nfactor = eta/minibatch.size();
		for(int l = 0; l < layers-1; l++){
			this.weights[l] = this.weights[l].minus( sum_w[l].times(nfactor) );
			this.biases[l] = this.biases[l].minus( sum_b[l].times(nfactor) );
			//sum_w[l].print(1,16);
			//sum_b[l].print(1,16);
		}
	}

	public Tuple backProp(Tuple data){
		Matrix[] grad_b = new Matrix[layers-1];
		Matrix[] grad_w = new Matrix[layers-1];
		for(int l = 0; l < layers-1; l++){
			grad_b[l] = new Matrix(this.biases[l].getRowDimension(), 1);
			grad_w[l] = new Matrix(this.weights[l].getRowDimension(), 
									this.weights[l].getColumnDimension());
		}

		Matrix[] activations = new Matrix[layers];
		Matrix[] zs = new Matrix[layers-1];
		// The first activations are the data input
		Matrix a = new Matrix(data.x.length, 1);
		for(int i = 0; i < data.x.length; i++)
			a.set(i, 0, data.x[i]);
		//sigmoid(this.weights[0].times(a).plus(this.biases[0])).print(1, 16);
		activations[0] = a;
		// Forward pass
		for(int l = 0; l < layers-1; l++){
			// Notes that the indices for weights and activations are staggered.
			Matrix z = this.weights[l].times(a).plus(this.biases[l]);
			zs[l] = z;
			a = sigmoid(z);
			activations[l+1] = a;
		}
		//activations[activations.length-1].print(1,16);
		// Back propagation
		Matrix delta = hadamard( costPrime(activations[activations.length-1], data.y) , sigmoidPrime(zs[zs.length-1]) );
		//delta.print(1, 20);
		grad_b[grad_b.length-1] = delta;
		grad_w[grad_w.length-1] = delta.times(activations[activations.length-2].transpose());

		for(int l = grad_b.length-2; l >=0; l--){
			delta = hadamard( this.weights[l+1].transpose().times(delta) , sigmoidPrime(zs[l]) );
			//delta.print(1,20);
			grad_b[l] = delta;
			grad_w[l] = delta.times(activations[l].transpose()); // Remember that activations[l] really means [l-1]
		}

		Tuple res = new Tuple(grad_b, grad_w);
		return res;
	}

	public Matrix sigmoid(Matrix zs){
		Matrix res = new Matrix(zs.getRowDimension(), zs.getColumnDimension());
		for(int i = 0; i < zs.getRowDimension(); i++){
			double z = zs.get(i,0);
			double val = 1 / (1 + Math.exp(-z));
			res.set(i,0,val);
		}
		return res;
	}

	public Matrix sigmoidPrime(Matrix z){
		Matrix left = sigmoid(z);
		Matrix right = new Matrix(left.getRowDimension(), left.getColumnDimension(), 1);
		right = right.minusEquals(left);
		return left.arrayTimes(right);
	}

	public Matrix costPrime(Matrix a, double[] y){
		// First convert y to Matrix
		Matrix ym = new Matrix(y.length, 1);
		for(int i = 0; i < y.length; i++)
			ym.set(i,0,y[i]);
		// Note that a and y are column vectors
		return a.minus(ym);
	}

	// Hadamard/Schur wrapper for legibility
	public Matrix hadamard(Matrix a, Matrix b){
		return a.arrayTimes(b);
	}

	public int argmax(double[] y){
		int res = 0;
		double max = y[0];
		for(int i = 1; i < y.length; i++){
			if(y[i] > max){
				max = y[i];
				res = i;
			}
		}
		return res;
	}

	public double[] rescale(double[] arr, double xmin, double xmax, double a, double b){
		// rescale over [a,b]
		double[] res = new double[arr.length];
		for(int i = 0; i < arr.length; i++){
			res[i] = (b-a)*((arr[i] - xmin)/(xmax - xmin)) + a;
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

	public void readDataset(String filename, List<Tuple> data){
		// Retrieve data from mnist_train or mnist_test
		try{
			Scanner scanIn = new Scanner(new BufferedReader(new InputStreamReader(this.getClass().getResourceAsStream("/" + filename + ".csv"))));
			int i = 0;
			while(scanIn.hasNextLine()){
				String inputLine = scanIn.nextLine();
				String[] inArray = inputLine.split(",");
				double[] raw_x = new double[784];
				double[] raw_y = new double[10];
				for(int j = 0; j < inArray.length; j++){
					if(j == 0){
						int temp = Integer.parseInt(inArray[j]);
						raw_y = vectorize(temp, 10);
					}
					else raw_x[j - 1] = Double.parseDouble(inArray[j]);
				}
				raw_x = rescale(raw_x, 0, 255, 0, 1);
				data.add(i, new Tuple(raw_x, raw_y) );
				i++;
			}
		} catch (Exception e){
			System.out.println(e);
		}
	}
}