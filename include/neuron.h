#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron;

struct Dendrite {
  Neuron* from;
  Neuron* to;
  long double weight;
  long double value;
};

typedef std::vector<Dendrite*> NeuronDendrites;

class Neuron {
  public:
    Neuron();
    ~Neuron();

    void connectTo(Neuron* other);
    void forwardPropagate();
    void calculateDelta(long double expected);
    void calculateError(long double expected);
    void updateWeights(long double learningRate);

    // Should only be called if this neuron belongs to the input layer.
    void feed(long double value);
    long double getOutput();
    long double getError();

  private:
    long double sumInputsTimesWeights();
    long double sigmoid(long double value);
    long double calculateDeltaAverage();

  private:
    bool isInputNeuron;
    long double bias;
    long double output;
    long double delta;
    long double error;

    std::vector<long double> deltas;

    NeuronDendrites inputDendrites; // This class do not control the contents of its own input dendrites.
    NeuronDendrites outputDendrites;
};

#endif
