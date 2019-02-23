package com.numlern.app;
import java.io.*;
import java.util.*;


// Hidden layer class
public class Layer{

	public Neuron[] neurons;

	public Layer(int n, int nprev){
		// n is the number of neurons in the layer
		// nprev is the number of neurons in the previous layer
		this.neurons = new Neuron[n];

		for(int i = 0; i < this.neurons.length; i++){
			this.neurons[i] = new Neuron(nprev);
		}

	}

	public double[][] weights(){
		double [][] weights = new double[this.neurons.length][this.neurons[0].w.length];
		for (int j = 0; j < this.neurons.length; j++){
			for (int k = 0; k < this.neurons[j].w.length; k++)
				weights[j][k] = this.neurons[j].w[k];
		}
		return weights;
	}

	public double[] biases(){
		double[] biases = new double[this.neurons.length];
		for (int j = 0; j < this.neurons.length; j++)
			biases[j] = this.neurons[j].b;
		return biases;
	}

}