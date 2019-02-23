package com.numlern.app;
import java.io.*;
import java.util.*;


public class Neuron{
	public double [] w; // weights
	public double b; // bias

	public Neuron(int n){
		// n is the number of neurons
		// in the preceding layer
		// (i.e. the size of this neuron's weight vector)
		this.w = new double[n];
	}

	// // get w
	// public double w(int i){
	// 	return this.w[i];
	// }

	// // set w
	// public void w(int i, double val){
	// 	this.w[i] = val;
	// }

	// public double b(){
	// 	return this.b;
	// }

	// public void b(double val){
	// 	this.b = val;
	// }

}