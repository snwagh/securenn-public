
#pragma once
#include "ChameleonCNNConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

// extern int partyNum;
// extern void AES_random_shuffle(vector<smallType> &vec, size_t begin_offset, size_t end_offset);


class ChameleonCNNLayer : public Layer
{
private:
	ChameleonCNNConfig conf;
	vector<myType> activations;
	vector<myType> deltas;
	vector<myType> weights;
	vector<myType> biases;
	vector<smallType> reluPrimeSmall;
	vector<myType> reluPrimeLarge;


public:
	//Constructor and initializer
	ChameleonCNNLayer(ChameleonCNNConfig* conf);
	void initialize();

	//Functions
	void forward(const vector<myType>& inputActivation) override;
	void computeDelta(vector<myType>& prevDelta) override;
	void updateEquations(const vector<myType>& prevActivations) override;
	void findMax(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns) override;

	//Getters
	vector<myType>* getActivation() {return &activations;};
	vector<myType>* getDelta() {error("Chameleon backprop and all not implemented");};

private:
	void maxSA(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns);
	void maxMPC(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns);
};