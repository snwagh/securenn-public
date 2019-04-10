
#ifndef AESOBJECT_H
#define AESOBJECT_H

#pragma once
#include <algorithm>
#include "globals.h"


class AESObject
{
private:
	//AES variables
	__m128i pseudoRandomString[RANDOM_COMPUTE];
	__m128i tempSecComp[RANDOM_COMPUTE];
	unsigned long rCounter = -1;
	AES_KEY_TED aes_key;

	//Extraction variables
	__m128i randomBitNumber {0};
	uint8_t randomBitCounter = 0;
	__m128i random8BitNumber {0};
	uint8_t random8BitCounter = 0; 
	__m128i random64BitNumber {0};
	uint8_t random64BitCounter = 0;

	//Private extraction functions
	__m128i newRandomNumber();

	//Private helper functions
	smallType AES_random(int i);

public:
	//Constructor
	AESObject(char* filename);
	
	//Randomness functions
	myType get64Bits();
	smallType get8Bits();
	smallType getBit();

	//Other randomness functions
	smallType randModPrime();
	smallType randNonZeroModPrime();
	myType randModuloOdd();
	void AES_random_shuffle(vector<smallType> &vec, size_t begin_offset, size_t end_offset);
};



#endif