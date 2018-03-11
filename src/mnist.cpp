#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <stdlib.h>

#include "mnist.h"

using namespace std;

vector<MNISTSample> MNIST::load(string filename) {
  vector<MNISTSample> samples;
  samples.reserve(5000);

  ifstream infile(filename.c_str());

  string line;
  while (getline(infile, line)) {
    char *cstr = new char[line.length() + 1];
    strcpy(cstr, line.c_str());

    char* charTok = strtok(cstr, ",");

    // Get first value
    MNISTSample mnist;
    mnist.sample.reserve(784);
    mnist.expected = strtol(charTok, NULL, 0);

    charTok = strtok(NULL, ",");
    while(charTok) {
      long double value = strtol(charTok, NULL, 0);
      mnist.sample.push_back(value/255);
      charTok = strtok(NULL, ",");
    }

    samples.push_back(mnist);
    delete cstr[];
  }

  return samples;
}
