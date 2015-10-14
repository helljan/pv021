package com.mycompany.neuron;

import java.util.Random;

public class main {

    static Perceptron ptron;
    static Trainer[] training = new Trainer[2000];
    
    static float f(float x) {
        return 2 * x +  1;
    }
    
    static void setup() {
        ptron = new Perceptron(3);
        
        Random rand = new Random();
        for (int i = 0; i < training.length; i++) {
            float x = rand.nextFloat() * 2 - 1;
            float y = rand.nextFloat() * 2 - 1;
            int answer = 1;
            if (y < f(x)) answer = -1;
            training[i] = new Trainer(x, y, answer);
        }
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        setup();
        
        for (int i = 0; i < training.length - 10; i++) {
            ptron.train(training[i].inputs, training[i].answer);
        }
        
        for (int i = training.length - 10; i < training.length; i++) {
            int answer = ptron.feedForward(training[i].inputs);
            System.out.println(answer == training[i].answer);
        }
    }
    
}
