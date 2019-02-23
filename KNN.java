package com.numlern.app;
import java.io.*;
import java.util.*;
import Jama.*;

public class KNN{
	private int dimensions;
	private int k;
	private double[][] train_x;
	private double[][] train_x_reduced;
	private int[] train_y;
	private double[] means;
	private Matrix v_transpose_reduced;

	public KNN(){
		this.train_x = new double[60000][784];
		this.train_y = new int[60000];
		try{
			Scanner scanIn = new Scanner(new BufferedReader(new InputStreamReader(this.getClass().getResourceAsStream("/mnist_train.csv"))));

			int i = 0;
			while(scanIn.hasNextLine()){
				String inputLine = scanIn.nextLine();

				String[] inArray = inputLine.split(",");

				for(int j = 0; j < inArray.length; j++){
					if(j == 0){
						this.train_y[i] = Integer.parseInt(inArray[j]);
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
	}

	public void train(int k, int dimensions){
		this.dimensions = dimensions;
		this.k = k;

		// PCA method does not currently work (29% accuracy... very good with 8s???)

		/*
		this.train_x_reduced = new double[60000][dimensions];

		double[][] temp = new double[60000][784];
		//Copying this.train_x values to this.train_x_reduced
		for(int i = 0; i < 784; i++){
			for(int j = 0; j < 60000; j++){
				temp[j][i] = this.train_x[j][i];
			}
		}

		// Normalizing
		this.means = new double[784];

		for(int i = 0; i < 784; i++){
			int sum = 0;
			for(int j = 0; j < 60000; j++){
				sum += temp[j][i];
			}

			this.means[i] = sum / 60000;
		}

		for(int i = 0; i < 784; i++){
			for(int j = 0; j < 60000; j++){
				temp[j][i] -= means[i];
			}
		}

		Matrix x = new Matrix(temp);

		//SingularValueDecomposition svd = x.svd();

		//Matrix v = svd.getV();

		//Getting v already computed since it takes about 40 minutes
		double[][] v = new double[784][784];

		try{
			Scanner scanIn = new Scanner(new BufferedReader(new FileReader("V.csv")));

			int i = 0;
			while(scanIn.hasNextLine()){
				String inputLine = scanIn.nextLine();

				String[] inArray = inputLine.split(",");

				for(int j = 0; j < inArray.length; j++){
					v[i][j] = Double.parseDouble(inArray[j]);
				}

				i++;
			}
		}catch (Exception e){
			System.out.println(e);
		}

		Matrix v_matrix = new Matrix(v);

		double[][] v_transpose = v_matrix.transpose().getArray();

		double[][] v_transpose_l = new double[784][dimensions];

		for(int i = 0; i < 784; i++){
			for(int j = 0; j < dimensions; j++){
				v_transpose_l[i][j] = v_transpose[i][j];
			}
		}

		this.v_transpose_reduced = new Matrix(v_transpose_l);

		Matrix t = x.times(v_transpose_reduced);

		this.train_x_reduced = t.getArray();*/
	}

	public int predict(double[] vector){
		for(int i = 0; i < 784; i++){
			vector[i] *= 255;
			//vector[i] -= this.means[i];
		}
		/*
		double[][] input = new double[1][784];

		input[0] = vector;

		Matrix input_vector = new Matrix(input);

		Matrix result = input_vector.times(this.v_transpose_reduced);

		double[] result_arr = result.getArray()[0];*/

		Tuple[] distances = new Tuple[60000];

		for(int i = 0; i < 60000; i ++){
			double sum = 0;
			for(int j = 0; j < this.dimensions; j++){
				//sum += Math.pow(result_arr[j] - this.train_x_reduced[i][j], 2);
				sum += Math.pow(vector[j] - this.train_x[i][j], 2);
			}

			double distance = Math.sqrt(sum);

			distances[i] = new Tuple(distance, this.train_y[i]);
		}

		// Sorting all of the distances
		Arrays.sort(distances);

		int[] counts = new int[10];
		// Considering the max of the K-nearest neighbors
		for(int i = 0; i < this.k; i++){
			counts[distances[i].getLabel()] ++;
		}

		int maxCount = counts[0];
		int label = 0;
		for(int i = 1; i < counts.length; i++){
			if(counts[i] > maxCount){
				label = i;
			}
		}

		return label;
	}

	public void test(int testsize){
		if (testsize > 10000) testsize = 10000;
		if (testsize < 1) testsize = 1;
		double[][] test_x = new double[10000][784];
		int[] test_y = new int[10000];
		System.out.println("Testing KNN on " + testsize + " data points...");
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

		int count = 0;
		for(int i = 0; i < testsize; i++){
			if(this.predict(test_x[i]) == test_y[i]){
				count++;
			}
			System.out.println(i + "/" + testsize);
		}
		System.out.println( ( (double) count / testsize));
	}
}