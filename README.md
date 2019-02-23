### Jamiel Rahi and Arnaud L'Heureux  
#### November 2018  

# Handwritten-Digit Classifier

## What works:

* Naive Bayes
* Simple KNN
* Convolution filter

## What doesn't work:

* Neural Net implementation 1
* Neural Net implementation 2 (simpler)
* Dimensionality reduction with PCA for KNN


## Dependencies:

* JAMA library for matrix operations
* Spark library to create a server
* The MNIST dataset

## Notes:

* tools.js includes a convolution algorithm and other matrix algorithms
* The original implementation works for small datasets (up to about 50 datapoints).
* The "alternate" folder contains a rewritten neural net using the Jama library for matrix operations (instead of my own functions). The goal was to try something closer to what has been done before as a sanity-check, and also reduce the number of lines of code. It turns out it works even worse than the original implementation.

