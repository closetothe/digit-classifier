
// Simple average kernel
// var k = [[1/9, 1/9, 1/9],
// 		 [1/9, 1/9, 1/9], 
// 		 [1/9, 1/9, 1/9],]


var pixels = new Array(28*8);
console.log("hello")

function canvasTo1DBinary(input,output){
	// Turn colored canvas image into binary image
	// expects: input = ctx.getImageData(0,0,w,h).data;
	// output -> assumes empty 1D array

	for(var i = 0; i < input.length-3; i+=4){
		// Alpha channel is between 0 or 255
		if(input[i+3] > 0) output.push(1);
		else output.push(0);
		// for floats: output.push(input[i+3]/255);
	}
}

function dimensionUp(input, w, output){
	// output -> assumes empty array
	while(input.length > 0) {
		output.push(input.splice(0,w));
	}
}

function dimensionDown(input, output){
	for(var i = 0; i < input.length; i++){
		for(var j = 0; j < input[i].length; j++){
			output.push(input[i][j]);
		}
	}
}

function binaryToCanvas(input, w, h, ctx){
	// input: 1D binary
	// output: 1D RGBA data
	
	var imgData = ctx.createImageData(w,h);

	for(var i = 0; i < imgData.data.length; i++){
		imgData.data[i] = 0;
	}
	console.log(imgData);
	for(var i = 0; i < input.length; i++){
		// Alpha channel is always 0 or 255
		// Fill all rgb values with 0 and alpha with 255
		// (black)
		imgData.data[(i+1)*4-1] = input[i]*255;
	}
	
	return imgData;
	
}

function convolution(input, output, k, step){
	//input -> 2D binary array
	//output -> 1D binary array of smaller size
	//k -> 2D kernel
	for(var i = 0; i < input.length-step+1; i+=step){
		for(var j = 0; j < input[i].length-step+1; j+=step){
			var dot = 0;
			for(var ki = 0 ; ki < k.length; ki++){
				for(var kj = 0 ; kj < k[ki].length; kj++){
					dot += k[ki][kj]*input[i+ki][j+kj];
				}
			}

			output.push(dot);

		}
	}
}

function dotProduct2D(A,B){
	// Assuming 2D matrices of the same size
	var prod = 0;
	for(var i = 0; i < A.length-1; i++){
		for(var j = 0; j < A[i].length-1; j++){
			prod += A[i][j]*B[i][j];
		}
	}
	return prod;
}






