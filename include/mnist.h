#ifndef MNIST_H
#define MNIST_H

#include <string>
#include <vector>

struct MNISTSample {
  long double expected;
  std::vector<long double> sample;
};

class MNIST {
  public:
    static std::vector<MNISTSample> load(std::string filename);
};

#endif
