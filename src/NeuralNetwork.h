
#pragma once
#include "NeuralNetConfig.h"
#include "FCLayer.h"
#include "CNNLayer.h"
#include "tools.h"
#include "globals.h"
using namespace std;

class NeuralNetwork
{
public:
	vector<myType> inputData;
	vector<myType> outputData;
	vector<Layer*> layers;

	NeuralNetwork(NeuralNetConfig* config);
	~NeuralNetwork();
	void forward();
	void backward();
	void computeDelta();
	void updateEquations();
	void predict(vector<myType> &maxIndex);
	void getAccuracy(const vector<myType> &maxIndex, vector<size_t> &counter);
};