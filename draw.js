
/*
Canvas code help from:
http://www.williammalone.com/articles/create-html5-canvas-javascript-drawing-app/ 
*/

// Initialize 2D array
var px = [];
var py = [];

var pdrag = [];
var drawing = false;

// INPUT CANVAS
var canvasDiv = document.getElementById("canvas-div");
var context;
var c1 = {};
c1.color = "#000000";
c1.lineWidth = 14;
c1.lineJoin = "round";
c1.w = 28*8; // Dimensions
c1.h = c1.w;

// OUTPUT CANVAS
var pcanvasDiv = document.getElementById("pcanvas-div");
var pcontext;
var c2 = {};
c2.color = "#000000";
c2.lineWidth = 1;
c2.lineJoin = "square";
c2.w = 28; // Dimensions
c2.h = c2.w;

// 28x28 1D convolved array 
var input = [];

canvasInit(c1);
pcanvasInit(c2);


// EVENT LISTENERS
var cvs = $("#canvas");
cvs.mousedown(function(event){
	var x = event.pageX - this.offsetLeft;
	var y = event.pageY - this.offsetTop;
	drawing = true;
	savePixels(x, y, false);
	draw()
})

cvs.mousemove(function(event){
  if(drawing){
  	var x = event.pageX - this.offsetLeft;
  	var y = event.pageY - this.offsetTop;
    savePixels(x, y, true);
    draw();
  }
});
var k;
cvs.mouseup(function(){
  drawing = false;
  input = downsample();
  paint(input);

  // Send classification request
});


cvs.mouseleave(function(){
  drawing = false;
  input = downsample();
  paint(input);
  var str = "[ ";
  for(var i = 0; i < input.length; i++){
  	str += input[i] + ", "
  }
  str += "]";
  $("#a").html(str);
  // Send classification request
});

// FUNCTIONS

function canvasInit(args){
	// Create canvas
	// (Complicated way for IE support)
	canvas = document.createElement("canvas");
	canvas.setAttribute('width', args.w);
	canvas.setAttribute('height', args.h);
	canvas.setAttribute('id', 'canvas');
	canvasDiv.appendChild(canvas);
	if(typeof G_vmlCanvasManager != 'undefined') {
		canvas = G_vmlCanvasManager.initElement(canvas);
	}
	context = canvas.getContext("2d");
	context.strokeStyle = args.color;
	context.lineJoin = args.lineJoin;
	context.lineWidth = args.lineWidth;
}

// Not very DRY, I know...
// I was having problems with scope.

function pcanvasInit(args){
	// Create canvas
	// (Complicated way for IE support)
	pcanvas = document.createElement("canvas");
	pcanvas.setAttribute('width', args.w);
	pcanvas.setAttribute('height', args.h);
	pcanvas.setAttribute('id', 'pcanvas');
	pcanvasDiv.appendChild(pcanvas);
	if(typeof G_vmlCanvasManager != 'undefined') {
		pcanvas = G_vmlCanvasManager.initElement(pcanvas);
	}
	pcontext = pcanvas.getContext("2d");
	pcontext.strokeStyle = args.color;
	pcontext.lineJoin = args.lineJoin;
	pcontext.lineWidth = args.lineWidth;
}

function savePixels(x, y, dragging){
	px.push(x);
	py.push(y);
	pdrag.push(dragging);
}

function draw(){
	clearCanvas(context, false);
	for(var i = 0; i < px.length; i++){
		context.beginPath();

		// The draggin array allows for smooth dragging
		// instead of a bunch of dots

		if(pdrag[i]) context.moveTo(px[i-1], py[i-1]);
		else context.moveTo(px[i]-1, py[i]);
		context.lineTo(px[i], py[i]);
		context.closePath();
		context.stroke();
	}
}


function downsample(){
  var img = context.getImageData(0,0,c1.w,c1.h).data;
  var conv = [];
  canvasTo1DBinary(img, conv);

  var img3 = [];
  dimensionUp(conv, c1.w, img3); // conv is now empty
  // img3 is a 2D version of conv
  k = [];
  for (var i = 0; i < 8; i++){
  	k.push([1/64,1/64,1/64,1/64,1/64,1/64,1/64,1/64]);
  }
  // Reuse conv
  conv = [];
  console.log(img3.length);
  convolution(img3,conv,k,8);
  console.log(conv.length);	
  // img2 is now 28x28 convolved
  return conv;
}

function paint(array){
	// Paint downsampled 1D image to pcontext
	var imgData = binaryToCanvas(array, 28, 28, pcontext);
	pcontext.putImageData(imgData, 0, 0);
}

function clearCanvas(ctx, full){
	ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
	if (full){
		px = [];
		py = [];
	}
}
