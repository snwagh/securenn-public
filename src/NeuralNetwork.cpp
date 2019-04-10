
#pragma once
#include "NeuralNetwork.h"
using namespace std;



NeuralNetwork::NeuralNetwork(NeuralNetConfig* config)
:inputData(LAYER0 * MINI_BATCH_SIZE),
 outputData(LAST_LAYER_SIZE * MINI_BATCH_SIZE)
{
	for (size_t i = 0; i < NUM_LAYERS - 1; ++i)
	{
		if (config->layerConf[i]->type.compare("FC") == 0)
			layers.push_back(new FCLayer(config->layerConf[i]));
		else if (config->layerConf[i]->type.compare("CNN") == 0)
			layers.push_back(new CNNLayer(config->layerConf[i]));
		else if (config->layerConf[i]->type.compare("ChameleonCNN") == 0)
			layers.push_back(new CNNLayer(config->layerConf[i]));
		else
			error("Only FC, CNN and ChameleonCNN layer types currently supported");
	}
}


NeuralNetwork::~NeuralNetwork()
{
	for (vector<Layer*>::iterator it = layers.begin() ; it != layers.end(); ++it)
		delete (*it);

	layers.clear();
}


void NeuralNetwork::forward()
{
	log_print("NN.forward");

	layers[0]->forward(inputData);

	for (size_t i = 1; i < NUM_LAYERS - 1; ++i)
		layers[i]->forward(*(layers[i-1]->getActivation()));

	// if (PRIMARY)
	// 	funcReconstruct2PC(*(layers[LL]->getActivation()), (*(layers[LL]->getActivation())).size(), "LL: activations");
}

void NeuralNetwork::backward()
{
	log_print("NN.backward");

	computeDelta();
	// cout << "computeDelta done" << endl;
	updateEquations();
}

void NeuralNetwork::computeDelta()
{
	log_print("NN.computeDelta");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	size_t size = rows*columns;
	size_t index;

	vector<myType> rowSum(size, 0);
	vector<myType> quotient(size, 0);

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
			rowSum[i*columns] += (*(layers[LL]->getActivation()))[i * columns + j];

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
			rowSum[i*columns + j] = rowSum[i*columns];

//DIVISION CODE BEGINS HERE
	if (STANDALONE)
	{
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
			{
				index = i * columns + j;
				if (rowSum[index] != 0)
					quotient[index] = divideMyTypeSA((*(layers[LL]->getActivation()))[index], rowSum[index]);
			}
	}
	else
	{
		funcDivisionMPC(*(layers[LL]->getActivation()), rowSum, quotient, size);
	}
//DIVISION CODE ENDS HERE

//WITHOUT DIVISION BEGINS HERE
	// for (size_t i = 0; i < rows; ++i)
	// 	for (size_t j = 0; j < columns; ++j)
	// 		quotient[i * columns + j] += (*(layers[LL]->getActivation()))[i * columns + j];

	// if (STANDALONE)
	// 	for (size_t i = 0; i < quotient.size(); ++i)
	// 		quotient[i] *= reluPrimeSmall[i];

	// if (MPC)
	// 	error("Implement multiplication by reluPrime here");
//WITHOUT DIVISION ENDS HERE

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
		{
			index = i * columns + j;
			(*(layers[LL]->getDelta()))[index] = quotient[index] - outputData[index];
		}

	for (size_t i = LL; i > 0; --i)
		layers[i]->computeDelta(*(layers[i-1]->getDelta()));
}

void NeuralNetwork::updateEquations()
{
	log_print("NN.updateEquations");

	for (size_t i = LL; i > 0; --i)
		layers[i]->updateEquations(*(layers[i-1]->getActivation()));	

	layers[0]->updateEquations(inputData);
}

void NeuralNetwork::predict(vector<myType> &maxIndex)
{
	log_print("NN.predict");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	vector<myType> max(rows);

	layers[LL]->findMax(*(layers[LL]->getActivation()), max, maxIndex, rows, columns);
}

void NeuralNetwork::getAccuracy(const vector<myType> &maxIndex, vector<size_t> &counter)
{
	log_print("NN.getAccuracy");

	size_t rows = MINI_BATCH_SIZE;
	size_t columns = LAST_LAYER_SIZE;
	vector<myType> max(rows), groundTruth(rows, 0);

	layers[LL]->findMax(outputData, max, groundTruth, rows, columns);

	if (STANDALONE)
	{
		for (size_t i = 0; i < MINI_BATCH_SIZE; ++i)
		{
			counter[1]++;
			if (maxIndex[i] == groundTruth[i])
				counter[0]++;
		}
	}
	else
	{
		//Reconstruct things
		vector<myType> temp_max(rows), temp_groundTruth(rows);
		if (partyNum == PARTY_B)
			sendTwoVectors<myType>(max, groundTruth, PARTY_A, rows, rows);

		if (partyNum == PARTY_A)
		{
			receiveTwoVectors<myType>(temp_max, temp_groundTruth, PARTY_B, rows, rows);
			addVectors<myType>(temp_max, max, temp_max, rows);
			dividePlainSA(temp_max, (1 << FLOAT_PRECISION));
			addVectors<myType>(temp_groundTruth, groundTruth, temp_groundTruth, rows);	
		}

		for (size_t i = 0; i < MINI_BATCH_SIZE; ++i)
		{
			counter[1]++;
			if (temp_max[i] == temp_groundTruth[i])
				counter[0]++;
		}		
	}

	cout << "Rolling accuracy: " << counter[0] << " out of " 
		 << counter[1] << " (" << (counter[0]*100/counter[1]) << " %)" << endl;
}


