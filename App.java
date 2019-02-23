package com.numlern.app;
import static spark.Spark.*;
import java.util.*;
import org.apache.log4j.*;

public class App 
{
    public static void main( String[] args )
    {

        staticFiles.location("/public");
        port(getHerokuAssignedPort());

    	//int[] sizes = new int[]{784, 30, 10};
    	//NN nn = new NN(sizes);
    	//nn.train(5, 3.0, 10);
    	//nn.test();
		

    	NaiveBayes naive = new NaiveBayes();
    	KNN knn = new KNN();
    	knn.train(6,784);
        knn.test(100);


    	get("/", (req, res) -> {
    		res.redirect("/index.html");
    		return "welcome...";
    	});

    	post("/naive", (req, res) -> {
    		String str = "" + req.body();
    		// System.out.println(str);
    		int[] input = toBinaryArray(parseInput(str));
    		// System.out.println(Arrays.toString(input));
    		int result = naive.predict(input);
    		res.type("text/xml");
    		return "{\"result\": \"" + result +"\"}";

    	});

    	post("/knn", (req, res) -> {
    		String str = "" + req.body();
    		// System.out.println(str);
    		double[] input = parseInput(str);
    		//System.out.println(Arrays.toString(input));
    		int result = knn.predict(input);
    		res.type("text/xml");
    		return "{\"result\": \"" + result +"\"}";

    	});

        // Change k
        post("/train/knn", (req, res) -> {
            String str = "" + req.body();
            // System.out.println(str);
            double[] input = parseInput(str);
            System.out.println(Arrays.toString(input));
            String result;
            if (input[0] > 0 && input[0] <= 784){
                knn.train((int)input[0], 784);
                result = "true";
            }
            else result = "false";
            res.type("text/xml");
            return "{\"result\": \"" + result +"\"}";

        });

        System.out.println( "Starting server..." );
    }


    public static double[] parseInput(String str){
    	// Accepts string of "[1,0,0,0.99238,...]"
    	double[] d = new double[784];

    	str = str.replace("[","");
    	str = str.replace("]","");
    	String [] split = str.split(",");

    	for(int i = 0; i < split.length; i++) {
    		d[i] = Double.parseDouble(split[i]);
    	}

    	//System.out.println(Arrays.toString(d));
    	return d;
    }

    public static int[] toBinaryArray(double[] d){

    	int[] b = new int[784];
		for(int i = 0; i < d.length; i++){
			if (d[i] >= 0.5) b[i] = 1;
			else b[i] = 0;
		}
		return b;

    }

    // Function provided by Spark documentation
    static int getHerokuAssignedPort() {
        ProcessBuilder processBuilder = new ProcessBuilder();
        if (processBuilder.environment().get("PORT") != null) {
            return Integer.parseInt(processBuilder.environment().get("PORT"));
        }
        return 8080;
    }
}














