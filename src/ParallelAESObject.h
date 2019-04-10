
#ifndef PARALLEL_AESOBJECT_H
#define PARALLEL_AESOBJECT_H

#pragma once
#include <algorithm>
#include "globals.h"

#define NONZERO_MAX 100
#define SHUFFLE_MAX 100
#define PC_CALLS_MAX 100



class ParallelAESObject
{
private:
	//Private helper functions
	smallType AES_random(int i, int t, int &offset);

	//precomputed random numbers
	smallType randomNumber[PC_CALLS_MAX*SHUFFLE_MAX*NO_CORES];
	smallType randomNonZero[PC_CALLS_MAX*NONZERO_MAX*NO_CORES];

	uint64_t counterPC = 0;

public:
	//Constructor
	ParallelAESObject(char* filename) {};
	precompute();

	//Other randomness functions
	smallType randNonZeroModPrime(int t, int &offset);
	void AES_random_shuffle(smallType *vec, size_t begin_offset, 
							size_t end_offset, int t, int &offset);
	void counterIncrement() {counterPC++;};
};



#endif