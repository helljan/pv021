package com.mycompany.neuron;

import java.util.Random;

public class Perceptron {
    float[] weights;
    float c = 0.01f;
    
    Perceptron(int n) {
        weights = new float[n];
        Random rand = new Random();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rand.nextFloat() * 2 - 1;
        }
    }
    
    int feedForward(float[] inputs) {
        float sum = 0;
        for (int i = 0; i < weights.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return activate(sum);
    }
    
    int activate(float sum) {
        if (sum > 0) return 1;
        else return -1;
    }
    
    void train(float[] inputs, int desired) {
        int guess = feedForward(inputs);
        float error = desired - guess;
        for (int i = 0; i < weights.length; i++) {
            weights[i] += c * error * inputs[i];
        }
    }
}
