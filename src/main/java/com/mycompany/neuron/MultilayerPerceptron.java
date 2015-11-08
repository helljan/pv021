package com.mycompany.neuron;

import java.util.ArrayList;
import java.util.List;

public class MultilayerPerceptron {
    List<List<Perceptron>> network;
    
    MultilayerPerceptron(List<Integer> numPerceptrons) {
        network = new ArrayList<>();
        
        network.add(createLayer(numPerceptrons.get(0), null, 0));
        for (int i = 1; i < numPerceptrons.size(); i++) {
            List<Perceptron> previousLayer = network.get(i - 1);
            network.add(createLayer(numPerceptrons.get(i), previousLayer, i));
        }
        
        //set outputs
        for (int i = 0; i < network.size() - 1; i++) {
            for (Perceptron perceptron : network.get(i)) {
                perceptron.outputs = network.get(i + 1);
            }
        }
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
    
    float[] getOutputs() {
        List<Perceptron> outputLayer = network.get(network.size() - 1);
        int numOutputPerceptrons = outputLayer.size();
        float[] result = new float[numOutputPerceptrons];
        for (int i = 0; i < numOutputPerceptrons; i++) {
            result[i] = outputLayer.get(i).y;
        }
        return result;
    }
    
    void train(List<float[]> inputs, List<float[]> outputs) {
        float epsilon = 0.5f;
        List<Perceptron> perceptrons = getAllPerceptrons(false);
        
        for (int t = 0; t < 10000; t++) {
            List<float[]> E_wji = gradient(inputs, outputs);
            for (int j = 0; j < perceptrons.size(); j++) {
                int numWeights = perceptrons.get(j).weights.length;
                for (int i = 0; i < numWeights; i++) {
                    perceptrons.get(j).weights[i] -= epsilon * E_wji.get(j)[i];
                }
            }
        }
    }
    
    private List<Perceptron> createLayer(int numPerceptrons, List<Perceptron> inputs, int layer) {
        List<Perceptron> list = new ArrayList<>();
        for (int i = 0; i < numPerceptrons; i++) {
            list.add(new Perceptron(inputs, layer, i));
        }
        return list;
    }
    
    private List<Perceptron> getAllPerceptrons(boolean inputLayer) {
        List<Perceptron> result = new ArrayList<>();
        for (int i = inputLayer ? 0 : 1; i < network.size(); i++) {
            result.addAll(network.get(i));
        }
        return result;
    }
    
    private List<float[]> gradient(List<float[]> inputs, List<float[]> outputs) {
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
            float[] Ek_yj = backpropagation(outputs.get(k));
            
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
        
        return Eji;
    }
    
    private float[] backpropagation(float[] outputs) {
        List<Perceptron> perceptrons = getAllPerceptrons(true);
        float[] Ek_yj = new float[perceptrons.size()];
        int numPerceptronsWithoutOutputLayer =
                perceptrons.size() - network.get(network.size() - 1).size();
        
        for (int j = perceptrons.size() - 1; j >= 0; j--) {
            if (perceptrons.get(j).outputs == null) {
                Ek_yj[j] = perceptrons.get(j).y -
                        outputs[j - numPerceptronsWithoutOutputLayer];
                perceptrons.get(j).Ek_yr = Ek_yj[j];
            } else {
                float sum = 0;
                for (int r = 0; r < perceptrons.get(j).outputs.size(); r++) {
                    sum += perceptrons.get(j).outputs.get(r).Ek_yr *
                            perceptrons.get(j).outputs.get(r).derivActivLogSig() *
                            perceptrons.get(j).outputs.get(r).weights[perceptrons.get(j).positionInLayer];
                }
            }
        }
        
        return Ek_yj;
    }
}
