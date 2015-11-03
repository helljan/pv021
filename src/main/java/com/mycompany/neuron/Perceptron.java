package com.mycompany.neuron;

import java.util.List;
import java.util.Random;

public class Perceptron {
    float[] weights;
    float c = 0.01f;
    float y = 0;
    final float lambda = 1;
    List<Perceptron> inputs;
    
    Perceptron(List<Perceptron> inputs) {
        if (inputs == null) {
            // this is input layer perceptron
        } else {
            this.inputs = inputs;
            weights = new float[inputs.size()];
            Random rand = new Random();
            for (int i = 0; i < weights.length; i++) {
                weights[i] = rand.nextFloat() * 2 - 1;
            }
        }
    }
    
    float feedForward() {
        float sum = 0;
        for (int i = 0; i < weights.length; i++) {
            sum += inputs.get(i).y * weights[i];
        }
        y = activateLogSig(sum);
        return y;
    }
    
    int activate(float sum) {
        if (sum > 0) return 1;
        else return -1;
    }
    
    float activateLogSig(float sum) {
        return (float) (1 / (1 + Math.exp(- lambda * sum)));
    }
    
    void train(int desired) {
        float guess = feedForward();
        float error = desired - guess;
        for (int i = 0; i < weights.length; i++) {
            weights[i] += c * error * inputs.get(i).y;
        }
    }
}
