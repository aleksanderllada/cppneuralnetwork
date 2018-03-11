#include "neuralnetwork.h"

#include <iostream>

using namespace std;

NeuralNetwork::NeuralNetwork(Mode mode)
  : mode(mode), learningRate(1.0), batchIterations(50), currentIteration(1) {
}

NeuralNetwork::~NeuralNetwork() {
  // Delete every neuron from every layer.
  deleteLayer(inputLayer);
  for (unsigned i = 0; i < hiddenLayers.size(); i++) {
    deleteLayer(hiddenLayers[i]);
  }
  deleteLayer(outputLayer);
}

void NeuralNetwork::deleteLayer(NeuralNetworkLayer& layer) {
  for (unsigned i = 0; i < layer.size; i++) {
    delete layer.neurons[i];
  }
}

void NeuralNetwork::setInputLayerSize(unsigned inputLayerSize) {
  inputLayer.size = inputLayerSize;
  inputLayer.neurons.reserve(inputLayerSize);
}

void NeuralNetwork::addHiddenLayer(unsigned hiddenLayerSize) {
  NeuralNetworkLayer newHiddenLayer;
  newHiddenLayer.size = hiddenLayerSize;
  newHiddenLayer.neurons.reserve(hiddenLayerSize);

  hiddenLayers.push_back(newHiddenLayer);
}

void NeuralNetwork::setOutputLayerSize(unsigned outputLayerSize) {
  outputLayer.size = outputLayerSize;
  outputLayer.neurons.reserve(outputLayerSize);
}

void NeuralNetwork::build() {
  // Create all layers (i.e. add neurons to each layer, according to its size)
  buildLayer(inputLayer);
  for (unsigned i = 0; i < hiddenLayers.size(); i++) {
    buildLayer(hiddenLayers[i]);
  }
  buildLayer(outputLayer);

  // Connect all layers
  connectLayer(inputLayer, hiddenLayers[0]);
  for (unsigned i = 0; i < hiddenLayers.size()-1; i++) {
    connectLayer(hiddenLayers[i], hiddenLayers[i+1]);
  }
  connectLayer(hiddenLayers[hiddenLayers.size()-1], outputLayer);
}

void NeuralNetwork::buildLayer(NeuralNetworkLayer& layer) {
  // Fill layer to all its capacity.
  for (unsigned i = 0; i < layer.size; i++) {
    layer.neurons.push_back(new Neuron());
  }
}

void NeuralNetwork::setBatchIterations(unsigned batchIterations) {
  this->batchIterations = batchIterations;
}

void NeuralNetwork::setLearningRate(long double learningRate) {
  this->learningRate = learningRate;
}

void NeuralNetwork::connectLayer(NeuralNetworkLayer& first, NeuralNetworkLayer& second) {
  for (unsigned i = 0; i < first.size; i++) {
    for (unsigned j = 0; j < second.size; j++) {
      first.neurons[i]->connectTo(second.neurons[j]);
    }
  }
}

long double NeuralNetwork::feed(long double expected, vector<long double> data) {
  if (data.size() != inputLayer.neurons.size()) {
    cout << "Invalid data size." << endl;
    return 0;
  }

  //cout << "Expected: " << expected << endl;
  // Feed input neurons and forward propagate the value.
  for (unsigned i = 0; i < inputLayer.size; i++) {
    inputLayer.neurons[i]->feed(data[i]);
  }

  // Forward propagate every neuron of every hidden layer.
  for (unsigned i = 0; i < hiddenLayers.size(); i++) {
    for (unsigned j = 0; j < hiddenLayers[i].size; i++) {
      hiddenLayers[i].neurons[j]->forwardPropagate();
    }
  }

  // Forward propagate every neuron from the output layer.
  for (unsigned i = 0; i < outputLayer.size; i++) {
    outputLayer.neurons[i]->forwardPropagate();
  }

  // Calculate the error
  long double error = 0.0;
  for (unsigned i = 0; i < outputLayer.size; i++) {
    outputLayer.neurons[i]->calculateError((i == expected) ? 1.0 : 0.0);
    error += outputLayer.neurons[i]->getError();
  }

  // Calculate delta (through backpropagation)
  for (unsigned i = 0; i < outputLayer.size; i++) {
    outputLayer.neurons[i]->calculateDelta((i == expected) ? 1.0 : 0.0);
  }

  for (int i = hiddenLayers.size()-1; i >= 0; i--) {
    for (unsigned j = 0; j < hiddenLayers[i].size; j++) {
      hiddenLayers[i].neurons[j]->calculateDelta(expected);
    }
  }

  // Update weights if stochastic or if batch is completed
  if (mode == Stochastic || (mode == Batch && currentIteration == batchIterations)) {
    for (unsigned i = 0; i < outputLayer.size; i++) {
      outputLayer.neurons[i]->updateWeights(learningRate);
    }

    for (unsigned i = 0; i < hiddenLayers.size(); i++) {
      for (unsigned j = 0; j < hiddenLayers[i].size; j++) {
        hiddenLayers[i].neurons[j]->updateWeights(learningRate);
      }
    }
  }

  if (mode == Batch) {
    if (currentIteration == batchIterations)
      currentIteration = 1;
    else
      currentIteration++;
  }

  return error;
}

void NeuralNetwork::updateWeights() {
  // Update weights after being fed
}

void NeuralNetwork::forwardPropagate() {
  // Propagate forwards
}

void NeuralNetwork::backPropagate() {
  // Propagate backwards
}
