package com.mycompany.neuron;

import java.util.Random;

public class main {

    Perceptron ptron;
    Trainer[] training = new Trainer[2000];
    int count = 0;
    
    float f(float x) {
        return 2 * x + 1;
    }
    
    void setup() {
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
        System.out.println("Hello world!");
    }
    
}
