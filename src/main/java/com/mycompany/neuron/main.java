package com.mycompany.neuron;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

public class main {

    static MultilayerPerceptron ptron;
    static List<float[]> inputs;
    static List<float[]> outputs;
    
    static float f(float x) {
        return 2 * x +  1;
    }
    
    static List<float[]> getInputs(String fileName)
            throws FileNotFoundException, IOException {
        Reader file = new FileReader(fileName);
        Iterable<CSVRecord> records = CSVFormat.DEFAULT.parse(file);
        List<float[]> result = new ArrayList<>();
        for (CSVRecord record : records) {
            float[] rec = new float[4];
            for (int i = 0; i < 4; i++) {
                rec[i] = Float.parseFloat(record.get(i));
            }
            result.add(rec);
        }
        return result;
    }
    
    static List<float[]> getOutputs(String fileName)
            throws IOException {
        Reader file = new FileReader(fileName);
        Iterable<CSVRecord> records = CSVFormat.DEFAULT.parse(file);
        List<float[]> result = new ArrayList<>();
        for (CSVRecord record : records) {
            float[] rec = new float[3];
            for (int i = 0; i < 3; i++) {
                rec[i] = 0;
            }
            switch (record.get(4)) {
                case "Iris-setosa":
                    rec[0] =  1;
                    break;
                case "Iris-versicolor":
                    rec[1] =  1;
                    break;
                case "Iris-virginica":
                    rec[2] =  1;
                    break;
            }
            result.add(rec);
        }
        return result;
    }
    
    static void setup() {
        List<Integer> layers = new ArrayList<>();
        layers.add(3);
        layers.add(1);
        ptron = new MultilayerPerceptron(layers);
        
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();
        
        Random rand = new Random();
        for (int i = 0; i < 2000; i++) {
            float x = rand.nextFloat() * 2 - 1;
            float y = rand.nextFloat() * 2 - 1;
            int answer = 1;
            if (y < f(x)) answer = -1;
            
            inputs.add(new float[]{x, y, 1});
            outputs.add(new float[]{answer});
        }
    }
    
    // Implementing Fisher–Yates shuffle
    static void shuffleArray(int[] ar)
    {
        // If running on Java 6 or older, use `new Random()` on RHS here
        Random rnd = ThreadLocalRandom.current();
        for (int i = ar.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            int a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }
    
    static void shuffleList(List<float[]> ar)
    {
        // If running on Java 6 or older, use `new Random()` on RHS here
        Random rnd = ThreadLocalRandom.current();
        for (int i = ar.size() - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            float[] a = ar.get(index);
            ar.set(index, ar.get(i));
            ar.set(i, a);
        }
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        //setup();
        
        List<Integer> layers = new ArrayList<>();
        layers.add(4);
        layers.add(4);
        layers.add(4);
        layers.add(3);
        ptron = new MultilayerPerceptron(layers);
        
        inputs = getInputs(args[0]);
        outputs = getOutputs(args[0]);
        
        ptron.train(inputs, outputs);
        
        for (int i = inputs.size() - 100; i < inputs.size(); i++) {
            ptron.feedForward(inputs.get(i));
            float[] answer = ptron.getOutputs();
            boolean right = true;
            float max = 0;
            int maxIndex = 0;
            float maxO = 0;
            for (int j = 0; j < 3; j++) {
                if (answer[j] > max) {
                    max = answer[j];
                    maxIndex = j;
                }
                if (outputs.get(i)[j] == 1) {
                    maxO = j;
                }
                System.out.print("" + answer[j] + "=" + outputs.get(i)[j] + ", ");
                if (answer[j] != outputs.get(i)[j])
                    right = false;
            }
            System.out.println(" " + (maxIndex == maxO));
        }
    }
    
}
