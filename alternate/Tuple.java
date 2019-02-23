package com.numlern.app;
import Jama.*;

public class Tuple implements Comparable<Tuple>{
	// KNN
	private double distance;
	private int label;

	// NN
	public Matrix[] b;
	public Matrix[] w;

	// NN -> data holder
	public double[] x;
	public double[] y;

	public Tuple(Matrix[] b, Matrix[] w){
		this.b = b;
		this.w = w;
	}

	public Tuple(double[] x, double[] y){
		this.x = x;
		this.y = y;
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