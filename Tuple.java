package com.numlern.app;

public class Tuple implements Comparable<Tuple>{
	// KNN
	private double distance;
	private int label;

	// NN minibatch
	public double[][] x;
	public double[][] y;

	// NN backprop
	public double[][] b;
	public double[][][] w;

	// Used in NN -> minibatch
	public Tuple(double[][]x, double[][]y){
		this.x = x;
		this.y = y;
	}

	// Used in NN -> back propagation

	public Tuple(double[][]b, double [][][]w){
		this.b = b;
		this.w = w;
	}	

	// Used in KNN
	public Tuple(double distance, int label){
		this.distance = distance;
		this.label = label;
	}

	// KNN
	public double getDistance(){
		return this.distance;
	}

	public int getLabel(){
		return this.label;
	}

	@Override
    public int compareTo(Tuple tuple) {
        // Comparaison selon la fréquence d'occurrence
        // Negatif => <, 0 => ==, Positif => >
        if((this.distance - tuple.distance) < 0){
        	return -1;
        }
        else if((this.distance - tuple.distance) > 0){
        	return 1;
		}
        else return 0;
    }

    @Override
    public String toString(){
    	return "" + distance + " : " + label;
    }
}