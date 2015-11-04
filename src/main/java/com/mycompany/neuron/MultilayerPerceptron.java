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
    
    List<Perceptron> getAllPerceptrons(boolean inputLayer) {
        List<Perceptron> result = new ArrayList<>();
        for (int i = inputLayer ? 0 : 1; i < network.size(); i++) {
            result.addAll(network.get(i));
        }
        return result;
    }
    
    int numWeights() {
        int result = 0;
        for (int i = 1; i < network.size(); i++) {
            result += network.get(i).size() * network.get(i - 1).size();
        }
        return result;
    }
    
    void train() {
        
    }
    
    void gradient(List<float[]> inputs, List<float[]> outputs) {
        List<Perceptron> perceptrons = getAllPerceptrons(false);
        
        // Eji = 0
        List<float[]> Eji = new ArrayList<>();
        for (int j = 0; j < perceptrons.size(); j++) {
            int numWeights = perceptrons.get(j).weights.length;
            Eji.add(new float[numWeights]);
            for (int i = 0; i < numWeights; i++) {
                Eji.get(j)[i] = 0;

            }
        }
        
        for (int k = 0; k < inputs.size(); k++) {
            // 1.
            feedForward(inputs.get(k));
            
            // 2.
            float[] Ek_yj = backpropagation();
            
            // 3.
            List<float[]> Ek_wji = new ArrayList<>();
            for (int j = 0; j < perceptrons.size(); j++) {
                int numWeights = perceptrons.get(j).weights.length;
                Ek_wji.add(new float[numWeights]);
                for (int i = 0; i < numWeights; i++) {
                    Ek_wji.get(j)[i] = Ek_yj[j] *
                            perceptrons.get(j).derivActivLogSig() *
                            perceptrons.get(j).inputs.get(i).y;
                            
                }
            }
            
            // 4.
            for (int j = 0; j < perceptrons.size(); j++) {
                int numWeights = perceptrons.get(j).weights.length;
                for (int i = 0; i < numWeights; i++) {
                    Eji.get(j)[i] += Ek_wji.get(j)[i];

                }
            }
        }
    }
    
    float[] backpropagation() {
        return new float[1];
    }
}
