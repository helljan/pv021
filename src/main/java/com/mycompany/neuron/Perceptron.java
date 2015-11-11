package com.mycompany.neuron;

import java.util.List;
import java.util.Random;

public class Perceptron {
    float[] weights;
    float c = 0.01f;
    float y = 0;
    final float lambda = 1;
    List<Perceptron> inputs;
    List<Perceptron> outputs;
    float Ek_yr = 0;
    int layer;
    int positionInLayer;
    
    Perceptron(List<Perceptron> inputs, int layer, int positionInLayer) {
        this.layer = layer;
        this.positionInLayer = positionInLayer;
        this.inputs = null;
        this.outputs = null;
        
        if (inputs == null) {
            // this is input layer perceptron
        } else {
            this.inputs = inputs;
            weights = new float[inputs.size() + 1];
            Random rand = new Random();
            for (int i = 0; i < weights.length; i++) {
                weights[i] = rand.nextFloat() * 2 - 1;
            }
        }
    }
    
    float feedForward() {
        float sum = 0;
        for (int i = 0; i < weights.length - 1; i++) {
            sum += inputs.get(i).y * weights[i];
        }
        sum += weights[weights.length - 1];
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
    
    float derivActivLogSig() {
        return lambda * y * (1 - y);
    }
    
    void train(int desired) {
        float guess = feedForward();
        float error = desired - guess;
        for (int i = 0; i < weights.length; i++) {
            weights[i] += c * error * inputs.get(i).y;
        }
    }
}
