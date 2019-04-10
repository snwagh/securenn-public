
#pragma once
#include "Functionalities.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;


class Layer
{
public: 
	virtual void forward(const vector<myType>& inputActivation) {};
	virtual void computeDelta(vector<myType>& prevDelta) {};
	virtual void updateEquations(const vector<myType>& prevActivations) {};
	virtual void findMax(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns) {};


//Getters
	virtual vector<myType>* getActivation() {};
	virtual vector<myType>* getDelta() {};
};