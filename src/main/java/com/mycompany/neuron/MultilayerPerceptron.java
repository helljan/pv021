package com.mycompany.neuron;

import java.util.ArrayList;
import java.util.List;

public class MultilayerPerceptron {
    List<List<Perceptron>> network;
    
    MultilayerPerceptron(List<Integer> numPerceptrons) {
        network = new ArrayList<>();
        
        network.add(createLayer(numPerceptrons.get(0), null));
        for (int i = 1; i < numPerceptrons.size(); i++) {
            List<Perceptron> previousLayer = network.get(i - 1);
            network.add(createLayer(numPerceptrons.get(i), previousLayer));
        }
    }
    
    private List<Perceptron> createLayer(int numPerceptrons, List<Perceptron> inputs) {
        List<Perceptron> list = new ArrayList<>();
        for (int i = 0; i < numPerceptrons; i++) {
            list.add(new Perceptron(inputs));
        }
        return list;
    }
    
    void feedForward(float[] inputs) {
        for (int i = 0; i < network.get(0).size(); i++) {
            network.get(0).get(i).y = inputs[i];
        }
        for (int i = 1; i < network.size(); i++) {
            for (Perceptron p: network.get(i)) {
                p.feedForward();
            }
        }
    }
    
    List<Perceptron> getAllPerceptrons() {
        List<Perceptron> result = new ArrayList<>();
        for (int i = 0; i < network.size(); i++) {
            result.addAll(network.get(i));
        }
        return result;
    }
    
    void train() {
        
    }
    
    void gradient(List<float[]> inputs, List<float[]> outputs) {
        List<Perceptron> perceptrons = getAllPerceptrons();
        
        float[] Eji = new float[perceptrons.size()];
        
        for (int k = 0; k < inputs.size(); k++) {
            // 1.
            feedForward(inputs.get(k));
            
            // 2.
            
            // 3.
            
        }
    }
    
    void backpropagation() {
        
    }
}
