
#pragma once
#include "LayerConfig.h"
#include "FCConfig.h"
#include "CNNConfig.h"
#include "ChameleonCNNConfig.h"
#include "globals.h"
using namespace std;

class NeuralNetConfig
{
public:
	size_t numIterations = 0;
	size_t numLayers = 0;
	vector<LayerConfig*> layerConf;

	NeuralNetConfig(size_t _numIterations)
	:numIterations(_numIterations)
	{};

	addLayer(FCConfig* fcl) {layerConf.push_back(fcl);};
	addLayer(CNNConfig* cnnl) {layerConf.push_back(cnnl);};
	addLayer(ChameleonCNNConfig* ccnnl) {layerConf.push_back(ccnnl);};
	
	checkNetwork() 
	{
		assert(layerConf.size() == (NUM_LAYERS-1) && "NeuralNetConfig is incorrect");
		assert(layerConf.back()->type.compare("FC") == 0 && "Last layer has to be FC");
		assert(((FCConfig*)layerConf.back())->outputDim == LAST_LAYER_SIZE && "Last layer size does not match LAST_LAYER_SIZE");
	};
};
