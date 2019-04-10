
#pragma once
#include "FCLayer.h"
using namespace std;

FCLayer::FCLayer(FCConfig* conf)
:conf(conf->batchSize, conf->inputDim, conf->outputDim),
 activations(conf->batchSize * conf->outputDim), 
 zetas(conf->batchSize * conf->outputDim), 
 deltas(conf->batchSize * conf->outputDim),
 weights(conf->inputDim * conf->outputDim),
 biases(conf->outputDim),
 reluPrimeSmall(conf->batchSize * conf->outputDim),
 reluPrimeLarge(conf->batchSize * conf->outputDim)
{
	initialize();
}


void FCLayer::initialize()
{
	//Initialize weights and biases here.
	//Ensure that initialization is correctly done.
	size_t lower = 30;
	size_t higher = 50;
	size_t decimation = 10000;
	size_t size = weights.size();

	vector<myType> temp(size);
	for (size_t i = 0; i < size; ++i)
		temp[i] = floatToMyType((float)(rand() % (higher - lower) + lower)/decimation);

	if (partyNum == PARTY_S)
		for (size_t i = 0; i < size; ++i)
			weights[i] = temp[i];
	else if (partyNum == PARTY_A or partyNum == PARTY_D)
		for (size_t i = 0; i < size; ++i)
			weights[i] = temp[i];
	else if (partyNum == PARTY_B or partyNum == PARTY_C)		
		for (size_t i = 0; i < size; ++i)
			weights[i] = 0;
		
	
	fill(biases.begin(), biases.end(), 0);
}



void FCLayer::forward(const vector<myType> &inputActivation)
{
	log_print("FC.forward");

	if (STANDALONE)
		forwardSA(inputActivation);
	else
		forwardMPC(inputActivation);
}

void FCLayer::computeDelta(vector<myType>& prevDelta)
{
	log_print("FC.computeDelta");

	if (STANDALONE)
		computeDeltaSA(prevDelta);
	else
		computeDeltaMPC(prevDelta);	
}

// void FCLayer::computeDelta(vector<myType>& prevDelta, const vector<myType>& prevReluPrimeLarge)
// {
// 	assert(THREE_PC && "computeDelta called incorrectly");

// 	computeDeltaMPC(prevDelta, prevReluPrimeLarge);	
// }


void FCLayer::updateEquations(const vector<myType>& prevActivations)
{
	log_print("FC.updateEquations");

	if (STANDALONE)
		updateEquationsSA(prevActivations);
	else
		updateEquationsMPC(prevActivations);
}


void FCLayer::findMax(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns)
{
	log_print("FC.findMax");

	if (STANDALONE)
		maxSA(a, max, maxIndex, rows, columns);
	else
		maxMPC(a, max, maxIndex, rows, columns);
}





/******************************** Standalone ********************************/
void FCLayer::forwardSA(const vector<myType> &inputActivation)
{
	//zetas = inputActivation * weights + biases
	//activations = ReLU(zetas)
	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t index;

	//Matrix Multiply
	matrixMultEigen(inputActivation, weights, zetas, 
					rows, common_dim, columns, 0, 0);
	dividePlainSA(zetas, (1 << FLOAT_PRECISION));

	//Add biases and ReLU
	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
		{
			index = i*columns + j; 

			zetas[index] += biases[j];
			reluPrimeSmall[index] = (zetas[index] < LARGEST_NEG ? 1:0);
			activations[index] = reluPrimeSmall[index]*zetas[index];
		}
}


void FCLayer::computeDeltaSA(vector<myType>& prevDelta)
{
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t common_dim = conf.outputDim;
	size_t tempSize = rows*common_dim;

	vector<myType> temp(tempSize);
	for (int i = 0; i < tempSize; ++i)
		temp[i] = deltas[i] * reluPrimeSmall[i];

	matrixMultEigen(temp, weights, prevDelta, 
					rows, common_dim, columns, 0, 1);

	dividePlainSA(prevDelta, (1 << FLOAT_PRECISION));
}


void FCLayer::updateEquationsSA(const vector<myType>& prevActivations)
{
	//Update Bias
	myType sum;

	for (size_t i = 0; i < conf.outputDim; ++i)
	{
		sum = 0;
		for (size_t j = 0; j < conf.batchSize; ++j)	
			sum += deltas[j * conf.outputDim + i];

		biases[i] -= dividePlainSA(multiplyMyTypesSA(sum, LEARNING_RATE, FLOAT_PRECISION), MINI_BATCH_SIZE);
	}

	//Update Weights
	size_t rows = conf.inputDim;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.batchSize;
	vector<myType> deltaWeight(rows * columns);

	matrixMultEigen(prevActivations, deltas, deltaWeight, 
					rows, common_dim, columns, 1, 0);
	dividePlainSA(deltaWeight, (1 << FLOAT_PRECISION));

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
			weights[i*columns + j] -= dividePlainSA(multiplyMyTypesSA(deltaWeight[i*columns + j], LEARNING_RATE, FLOAT_PRECISION), MINI_BATCH_SIZE);
}


//Chunk wise maximum of a vector of size rows*columns and max is caclulated of every 
//column number of elements. max is a vector of size rows. maxIndex contains the index of 
//the maximum value.
void FCLayer::maxSA(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns)
{
	size_t size = rows*columns;
	vector<myType> diff(size);

	for (size_t i = 0; i < rows; ++i)
	{
		max[i] = a[i*columns];
		maxIndex[i] = 0;
	}

	for (size_t i = 1; i < columns; ++i)
		for (size_t j = 0; j < rows; ++j)
		{
			if (a[j*columns + i] > max[j])
			{
				max[j] = a[j*columns + i];
				maxIndex[j] = i;
			}
		}
}



/******************************** MPC ********************************/
void FCLayer::forwardMPC(const vector<myType> &inputActivation)
{
	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows*columns;

	//Matrix Multiplication
	funcMatMulMPC(inputActivation, weights, zetas, 
				  rows, common_dim, columns, 
				  0, 0);

	//Add Biases
	if (PRIMARY)
		for(size_t r = 0; r < rows; ++r)
			for(size_t c = 0; c < columns; ++c)
				zetas[r*columns + c] += biases[c];

	if (FOUR_PC)
	{
		funcRELUPrime4PC(zetas, reluPrimeSmall, size);
		funcSelectShares4PC(zetas, reluPrimeSmall, activations, size);
	}

	if (THREE_PC)
	{
		funcRELUPrime3PC(zetas, reluPrimeLarge, size);
		funcSelectShares3PC(zetas, reluPrimeLarge, activations, size);
	}

	// if (PRIMARY)
	// 	funcReconstruct2PC(activations, DEBUG_CONST, "activations");
}


void FCLayer::computeDeltaMPC(vector<myType>& prevDelta)
{
	//Back Propagate	
	size_t rows = conf.batchSize;
	size_t columns = conf.inputDim;
	size_t common_dim = conf.outputDim;
	size_t size = rows*columns;
	size_t tempSize = rows*common_dim;

	//Since delta and weights are both unnaturally shared, modify into temp
	vector<myType> temp(tempSize);
	for (size_t i = 0; i < tempSize; ++i)
		temp[i] = deltas[i];

	if (FOUR_PC)
		funcSelectShares4PC(temp, reluPrimeSmall, temp, tempSize);

	if (THREE_PC)
		funcSelectShares3PC(temp, reluPrimeLarge, temp, tempSize);

	funcMatMulMPC(temp, weights, prevDelta, 
				  rows, common_dim, columns, 0, 1);
}

// void FCLayer::computeDeltaMPC(vector<myType>& prevDelta, const vector<myType>& prevReluPrimeLarge)
// {
// 	assert(THREE_PC && "computeDeltaMPC called in 3PC (incorrect) mode");

// 	//Back Propagate	
// 	size_t rows = conf.batchSize;
// 	size_t columns = conf.inputDim;
// 	size_t common_dim = conf.outputDim;
// 	size_t size = rows*columns;
// 	size_t tempSize = rows*common_dim;

// 	vector<myType> temp(tempSize);
// 	for (size_t i = 0; i < tempSize; ++i)
// 		temp[i] = deltas[i];

// 	funcSelectShares3PC(temp, reluPrimeLarge, temp, tempSize);
// 	funcMatMulMPC(temp, weights, prevDelta, 
// 				  rows, common_dim, columns, 0, 1);	

// 	// funcMatMulMPC(deltas, weights, prevDelta, 
// 	// 			  rows, common_dim, columns, 0, 1);
// 	// funcSelectShares3PC(prevDelta, prevReluPrimeLarge, prevDelta, size);
// }


void FCLayer::updateEquationsMPC(const vector<myType>& prevActivations)
{
	size_t rows = conf.batchSize;
	size_t columns = conf.outputDim;
	size_t common_dim = conf.inputDim;
	size_t size = rows*columns;	
	vector<myType> temp(columns, 0);

	for (size_t i = 0; i < rows; ++i)
		for (size_t j = 0; j < columns; ++j)
			temp[j] += deltas[i*columns + j];

	if (PRIMARY)
		funcTruncate2PC(temp, LOG_MINI_BATCH + LOG_LEARNING_RATE, columns, PARTY_A, PARTY_B);

	subtractVectors<myType>(biases, temp, biases, columns);


	//Update Weights 
	rows = conf.inputDim;
	columns = conf.outputDim;
	common_dim = conf.batchSize;
	size = rows*columns;
	vector<myType> deltaWeight(size);

	funcMatMulMPC(prevActivations, deltas, deltaWeight, 
   				  rows, common_dim, columns, 1, 0);

	if (PRIMARY)
		funcTruncate2PC(deltaWeight, LOG_MINI_BATCH + LOG_LEARNING_RATE, size, PARTY_A, PARTY_B);

	subtractVectors<myType>(weights, deltaWeight, weights, size);		
}


void FCLayer::maxMPC(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns)
{
	funcMaxMPC(a, max, maxIndex, rows, columns);
}

