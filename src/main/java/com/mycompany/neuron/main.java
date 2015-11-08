package com.mycompany.neuron;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class main {

    static MultilayerPerceptron ptron;
    static List<float[]> inputs;
    static List<float[]> outputs;
    
    static float f(float x) {
        return 2 * x +  1;
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
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        setup();
        
        ptron.train(inputs, outputs);
        
        for (int i = inputs.size() - 10; i < inputs.size(); i++) {
            ptron.feedForward(inputs.get(i));
            float[] answer = ptron.getOutputs();
            System.out.println("" + answer[0] + " " + outputs.get(i)[0]);
        }
    }
    
}
