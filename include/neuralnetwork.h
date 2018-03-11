#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

#include "neuron.h"

struct NeuralNetworkLayer {
  std::vector<Neuron*> neurons;
  unsigned size;
};

enum Mode {
  Stochastic,
  Batch
};

class NeuralNetwork {
  public:
    NeuralNetwork(Mode mode);
    ~NeuralNetwork();

    void setInputLayerSize(unsigned inputLayerSize);
    void addHiddenLayer(unsigned hiddenLayerSize);
    void setOutputLayerSize(unsigned outputLayerSize);

    void build();

    void setBatchIterations(unsigned batchIterations);
    void setLearningRate(long double learningRate);

    long double feed(long double expected, std::vector<long double> data);
    void updateWeights();

  private:
    void buildLayer(NeuralNetworkLayer& layer);
    void connectLayer(NeuralNetworkLayer& first, NeuralNetworkLayer& second);

    void forwardPropagate();
    void backPropagate();

    void deleteLayer(NeuralNetworkLayer& layer);

  private:
    Mode mode;
    long double learningRate;

    unsigned batchIterations;
    unsigned currentIteration;

    NeuralNetworkLayer inputLayer;
    std::vector<NeuralNetworkLayer> hiddenLayers;
    NeuralNetworkLayer outputLayer;
};

#endif
