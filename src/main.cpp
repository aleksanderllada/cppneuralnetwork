#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

#include "neuralnetwork.h"
#include "mnist.h"

using namespace std;

int main(int argc, char* argv[]) {
  srand(time(NULL));

  if (argc < 4) {
    cout << "Usage: neuralnetwork <stochastic|batch|minibatch> <hidden_units_count> <learning_rate> [mini_batch_size]" << endl;
    cout << "Ex1.:  neuralnetwork stochastic 25 1.0" << endl;
    cout << "Ex2.:  neuralnetwork batch 25 0.5" << endl;
    cout << "Ex3.:  neuralnetwork minibatch 25 10.0 50" << endl;

    return 0;
  }

  Mode mode;
  unsigned hiddenUnits;
  long double learningRate;
  unsigned miniBatchSize = 0;

  if (strncmp(argv[1], "stochastic", 10) == 0)
    mode = Stochastic;
  else if (strncmp(argv[1], "batch", 5) == 0)
    mode = Batch;
  else if (strncmp(argv[1], "minibatch", 9) == 0) {
    if (argc < 5) {
      cout << "Invalid parameters." << endl;
      return 0;
    }

    mode = Batch;
    miniBatchSize = atoi(argv[4]);
  } else {
    cout << "Invalid mode. Valid modes: <stochastic|batch|minibatch>." << endl;
    return 0;
  }

  hiddenUnits = atoi(argv[2]);
  learningRate = atof(argv[3]);

  NeuralNetwork network(mode);
  network.setLearningRate(learningRate); 

  cout << "Creating layers" << endl;
  network.setInputLayerSize(784);
  network.addHiddenLayer(hiddenUnits);
  network.setOutputLayerSize(10);

  cout << "Building the network" << endl;
  network.build();

  cout << "Loading the samples" << endl;
  vector<MNISTSample> mnistSamples = MNIST::load("data/data_tp1");

  if (miniBatchSize)
    network.setBatchIterations(miniBatchSize);
  else if (mode == Batch)
    network.setBatchIterations(mnistSamples.size());

  cout << "Training" << endl;
  for (unsigned epoch = 0; epoch < 1000; epoch++) {
    long double error = 0.0;
    for (unsigned i = 0; i < mnistSamples.size(); i++) {
      error += network.feed(mnistSamples[i].expected, mnistSamples[i].sample);
    }
    error /= 5000;
    cout << epoch << "," << error << endl;
  }

  return 0;
}
