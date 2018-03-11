#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "neuron.h"

using namespace std;

Neuron::Neuron()
  : isInputNeuron(false) {
    bias = ((long double)(rand() % 100000 + 1)) / 100000.0;
}

Neuron::~Neuron() {
  // Delete output dendrites.
  for (unsigned i = 0; i < outputDendrites.size(); i++) {
    delete outputDendrites[i];
  }
}

void Neuron::connectTo(Neuron* other) {
  Dendrite* newDendrite = new Dendrite();
  newDendrite->from = this;
  newDendrite->to = other;
  newDendrite->weight = ((long double)(rand() % 100000 + 1)) / 100000.0 ; // Initialize weights
  newDendrite->value = 0.0; // Initialize values

  this->outputDendrites.push_back(newDendrite);
  other->inputDendrites.push_back(newDendrite);
}

void Neuron::forwardPropagate() {
  if (!isInputNeuron) {
    // Calculate sum ( Wx * Ix)
    output = sumInputsTimesWeights() + bias;
    output = sigmoid(output); // Activation function
  }

  // Set all output dendrite values.
  for (unsigned i = 0; i < outputDendrites.size(); i++) {
    outputDendrites[i]->value = output;
  }
}

void Neuron::feed(long double value) {
  isInputNeuron = true;
  output = value;

  forwardPropagate();
}

void Neuron::calculateDelta(long double expected) {
  if (outputDendrites.size() == 0) {
    // Calculate output layer delta
    delta = output - expected;

  } else if (inputDendrites.size() > 0) {
    // Calculate hidden layer delta
    long double sum = 0.0;
    for (unsigned i = 0; i < outputDendrites.size(); i++) {
      sum += outputDendrites[i]->to->delta * outputDendrites[i]->weight;
    }

    delta = output - sum;
  }

  deltas.push_back(delta);
}

void Neuron::calculateError(long double expected) {
  error = -1 * expected * log (output) - (1 - expected) * log (1 - output);
}

void Neuron::updateWeights(long double learningRate) {
  delta = calculateDeltaAverage();

  bias -= delta * learningRate;
  for (unsigned i = 0; i < inputDendrites.size(); i++) {
    inputDendrites[i]->weight -= delta * learningRate * inputDendrites[i]->value;
  }

  deltas.clear();
}

long double Neuron::getOutput() {
  return output;
}

long double Neuron::getError() {
  return error;
}

long double Neuron::sumInputsTimesWeights() {
  long double sum = 0.0;

  for (unsigned i = 0; i < inputDendrites.size(); i++) {
    sum += inputDendrites[i]->weight * inputDendrites[i]->value;
  }

  return sum;
}

long double Neuron::sigmoid(long double value) {
  return 1 / (1 + exp(-value));
}

long double Neuron::calculateDeltaAverage() {
  double average = 0.0;

  for (unsigned i = 0; i < deltas.size(); i++) {
    average += deltas[i];
  }

  average /= deltas.size();

  return average;
}
