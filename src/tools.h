
#ifndef TOOLS_H
#define TOOLS_H
#pragma once

#include <stdio.h> 
#include <iostream>
#include "Config.h"
#include "../util/TedKrovetzAesNiWrapperC.h"
#include <wmmintrin.h>
#include <emmintrin.h>
#include <smmintrin.h>
#include <vector>
#include <time.h>
#include "secCompMultiParty.h"
#include "main_gf_funcs.h"
// #include "../util/sha256.h"
#include <string>
#include <openssl/sha.h>
#include <math.h>
#include <sstream>
#include "AESObject.h"
#include "ParallelAESObject.h"
#include "connect.h"
#include "globals.h"

extern int partyNum;

extern AESObject* aes_common;
extern AESObject* aes_indep;
extern AESObject* aes_a_1;
extern AESObject* aes_a_2;
extern AESObject* aes_b_1;
extern AESObject* aes_b_2;
extern AESObject* aes_c_1;
extern ParallelAESObject* aes_parallel;

extern smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
extern smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];



#if MULTIPLICATION_TYPE==0
#define MUL(x,y) gfmul(x,y)
#define MULT(x,y,ans) gfmul(x,y,&ans)

#ifdef OPTIMIZED_MULTIPLICATION
#define MULTHZ(x,y,ans) gfmulHalfZeros(x,y,&ans)//optimized multiplication when half of y is zeros
#define MULHZ(x,y) gfmulHalfZeros(x,y)//optimized multiplication when half of y is zeros
#else
#define MULTHZ(x,y,ans) gfmul(x,y,&ans)
#define MULHZ(x,y) gfmul(x,y)
#endif
#define SET_ONE _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)

#else 
#define MUL(x,y) gfmul3(x,y)
#define MULT(x,y,ans) gfmul3(x,y,&ans)
#ifdef OPTIMIZED_MULTIPLICATION
#define MULTHZ(x,y,ans) gfmul3HalfZeros(x,y,&ans)//optimized multiplication when half of y is zeros
#define MULHZ(x,y) gfmul3HalfZeros(x,y)//optimized multiplication when half of y is zeros
#else
#define MULTHZ(x,y,ans) gfmul3(x,y,&ans)
#define MULHZ(x,y) gfmul3(x,y)
#endif 
#define SET_ONE _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
#endif 

// 
//field zero
#define SET_ZERO _mm_setzero_si128()
//the partynumber(+1) embedded in the field
#define SETX(j) _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, j+1)//j+1
//Test if 2 __m128i variables are equal
#define EQ(x,y) _mm_test_all_ones(_mm_cmpeq_epi8(x,y))
//Add 2 field elements in GF(2^128)
#define ADD(x,y) _mm_xor_si128(x,y)
//Subtraction and addition are equivalent in characteristic 2 fields
#define SUB ADD
//Evaluate x^n in GF(2^128)
#define POW(x,n) fastgfpow(x,n)
//Evaluate x^2 in GF(2^128)
#define SQ(x) square(x)
//Evaluate x^(-1) in GF(2^128)
#define INV(x) inverse(x)
//Evaluate P(x), where p is a given polynomial, in GF(2^128)
#define EVAL(x,poly,ans) fastPolynomEval(x,poly,&ans)//polynomEval(SETX(x),y,z)
//Reconstruct the secret from the shares
#define RECON(shares,deg,secret) reconstruct(shares,deg,&secret)
//returns a (pseudo)random __m128i number using AES-NI
#define RAND LoadSeedNew
//returns a (pseudo)random bit using AES-NI
#define RAND_BIT LoadBool

//the encryption scheme
#define PSEUDO_RANDOM_FUNCTION(seed1, seed2, index, numberOfBlocks, result) pseudoRandomFunctionwPipelining(seed1, seed2, index, numberOfBlocks, result);

//The degree of the secret-sharings before multiplications
extern int degSmall;
//The degree of the secret-sharing after multiplications (i.e., the degree of the secret-sharings of the PRFs)
extern int degBig;
//The type of honest majority we assume
extern int majType;

//bases for interpolation
extern __m128i* baseReduc;
extern __m128i* baseRecon;
//saved powers for evaluating polynomials
extern __m128i* powers;

//one in the field
extern const __m128i ONE;
//zero in the field
extern const __m128i ZERO;

extern int testCounter;

typedef struct polynomial {
	int deg;
	__m128i* coefficients;
}Polynomial;


void gfmul(__m128i a, __m128i b, __m128i *res);

//This function works correctly only if all the upper half of b is zeros
void gfmulHalfZeros(__m128i a, __m128i b, __m128i *res);

//multiplies a and b
__m128i gfmul(__m128i a, __m128i b);

//This function works correctly only if all the upper half of b is zeros
__m128i gfmulHalfZeros(__m128i a, __m128i b);

__m128i gfpow(__m128i x, int deg);

__m128i fastgfpow(__m128i x, int deg);

__m128i square(__m128i x);

__m128i inverse(__m128i x);

string _sha256hash_(char *input, int length);

string sha256hash(char *input, int length);

void printError(string error);

string __m128i_toHex(__m128i var);

string __m128i_toString(__m128i var);

__m128i stringTo__m128i(string str);

unsigned int charValue(char c);

double diff(timespec start, timespec end);

string convertBooltoChars(bool *input, int length);

string toHex(string s);

string convertCharsToString(char *input, int size);

void print(__m128i* arr, int size);

void print128_num(__m128i var);




void print_usage(const char * bin);
void start_time();
void end_time(string str);
void start_rounds();
void end_rounds(string str);

void print_myType(myType var, string message, string type);
void print_linear(myType var, string type);


void matrixMultEigen(const vector<myType> &a, const vector<myType> &b, vector<myType> &c, 
					size_t rows, size_t common_dim, size_t columns,
				 	size_t transpose_a, size_t transpose_b);

myType divideMyTypeSA(myType a, myType b);
myType dividePlainSA(myType a, int b);
void dividePlainSA(vector<myType> &vec, int divisor);
myType multiplyMyTypesSA(myType a, myType b, int shift);
size_t partner(size_t party);
size_t adversary(size_t party);


inline smallType addModPrime(smallType a, smallType b)
{return additionModPrime[a][b];}
inline smallType multiplyModPrime(smallType a, smallType b)
{return multiplicationModPrime[a][b];}
smallType subtractModPrime(smallType a, smallType b);
inline smallType wrapAround(myType a, myType b)
{return (a > MINUS_ONE - b);}
void wrapAround(const vector<myType> &a, const vector<myType> &b, 
				vector<smallType> &c, size_t size);
void populateBitsVector(vector<smallType> &vec, string r_type, size_t size);
void sharesOfBits(vector<smallType> &bit_shares_x_1, vector<smallType> &bit_shares_x_2, 
				  const vector<myType> &x, size_t size, string r_type);
void sharesOfLSB(vector<smallType> &share_1, vector<smallType> &share_2, 
				  const vector<myType> &r, size_t size, string r_type);
void sharesOfLSB(vector<myType> &share_1, vector<myType> &share_2, 
				  const vector<myType> &r, size_t size, string r_type);
void sharesOfBitVector(vector<smallType> &share_1, vector<smallType> &share_2, 
				  const vector<smallType> &vec, size_t size, string r_type);
void sharesOfBitVector(vector<myType> &share_1, vector<myType> &share_2, 
				  const vector<smallType> &vec, size_t size, string r_type);
void splitIntoShares(const vector<myType> &a, vector<myType> &a1, vector<myType> &a2, size_t size);
void XORVectors(const vector<smallType> &a, const vector<smallType> &b, 
				  vector<smallType> &c, size_t size);
myType multiplyMyTypes(myType a, myType b, size_t shift);
void log_print(string str);
void error(string str);
void convolutionReshape(const vector<myType> &vec, vector<myType> &vecShaped,
						size_t ih, size_t iw, size_t C, size_t B,  
						size_t fh, size_t fw, size_t sy, size_t sx);
void maxPoolReshape(const vector<myType> &vec, vector<myType> &vecShaped,
						size_t ih, size_t iw, size_t D, size_t B,  
						size_t fh, size_t fw, size_t sy, size_t sx);
void convolutionReshapeBackprop(const vector<myType> &vec, vector<myType> &vecOut, 
								size_t imageH, size_t imageW, 
								size_t filterH, size_t filterW, 
								size_t strideY, size_t strideX, 
								size_t C, size_t D, size_t B);


void start_m();
void end_m(string str);





// Template functions
template<typename T>
void populateRandomVector(vector<T> &vec, size_t size,  string r_type, string neg_type);

template<typename T>
void addVectors(const vector<T> &a, const vector<T> &b, vector<T> &c, size_t size);

template<typename T>
void subtractVectors(const vector<T> &a, const vector<T> &b, vector<T> &c, size_t size);

template<typename T>
void copyVectors(const vector<T> &a, vector<T> &b, size_t size);

template<typename T1, typename T2>
void addModuloOdd(const vector<T1> &a, const vector<T2> &b, vector<myType> &c, size_t size);

template<typename T1, typename T2>
void subtractModuloOdd(const vector<T1> &a, const vector<T2> &b, vector<myType> &c, size_t size);

template<typename T1, typename T2>
myType addModuloOdd(T1 a, T2 b);

template<typename T1, typename T2>
myType subtractModuloOdd(T1 a, T2 b);

template<typename T>
void sharesModuloOdd(vector<myType> &shares_1, vector<myType> &shares_2, 
				  const vector<T> &x, size_t size, string r_type);

//Randmoization is passed as an argument here.
template<typename T>
void getVectorfromPrimary(vector<T> &vec, size_t size, string r_mode, string n_mode);


















//Template implementations
// template<typename T>
// void populateRandomVector(vector<T> &vec, size_t size, string r_type, string neg_type)
// {	
// 	assert((r_type == "COMMON" or r_type == "INDEP") && "invalid randomness type for populateRandomVector");
// 	assert((neg_type == "NEGATIVE" or neg_type == "POSITIVE") && "invalid negativeness type for populateRandomVector");
// 	assert(sizeof(T) == sizeof(myType) && "Probably only need 64-bit numbers");
// 	assert(r_type == "COMMON" && "Only common randomness mode required currently");
// }

template<typename T>
void populateRandomVector(vector<T> &vec, size_t size, string r_type, string neg_type)
{	
	// assert((r_type == "COMMON" or r_type == "INDEP") && "invalid randomness type for populateRandomVector");
	assert((neg_type == "NEGATIVE" or neg_type == "POSITIVE") && "invalid negativeness type for populateRandomVector");
	// assert(sizeof(T) == sizeof(myType) && "Probably only need 64-bit numbers");
	// assert(r_type == "COMMON" && "Only common randomness mode required currently");

	myType sign = 1;
	if (r_type == "COMMON")
	{
		if (neg_type == "NEGATIVE")
		{		
			if (partyNum == PARTY_B or partyNum == PARTY_D)
				sign = MINUS_ONE;

			if (sizeof(T) == sizeof(myType))
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = sign*aes_common->get64Bits();		
			}
			else
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = sign*aes_common->get8Bits();		
			}
		}
		
		if (neg_type == "POSITIVE")
		{
			if (sizeof(T) == sizeof(myType))
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = aes_common->get64Bits();		
			}
			else
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = aes_common->get8Bits();		
			}			
		}
	}

	if (r_type == "INDEP")
	{
		if (neg_type == "NEGATIVE")
		{		
			if (partyNum == PARTY_B or partyNum == PARTY_D)
				sign = MINUS_ONE;

			if (sizeof(T) == sizeof(myType))
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = sign*aes_indep->get64Bits();		
			}
			else
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = sign*aes_indep->get8Bits();		
			}		
		}
		
		if (neg_type == "POSITIVE")
		{
			if (sizeof(T) == sizeof(myType))
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = aes_indep->get64Bits();		
			}
			else
			{
				for (size_t i = 0; i < size; ++i)
					vec[i] = aes_indep->get8Bits();		
			}		
		}
	}

	if (r_type == "a_1")
	{
		assert((partyNum == PARTY_A or partyNum == PARTY_C) && "Only A and C can call for a_1");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(myType) && "sizeof(T) == sizeof(myType)");
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_a_1->get64Bits();
	}

	if (r_type == "b_1")
	{
		assert((partyNum == PARTY_A or partyNum == PARTY_C) && "Only A and C can call for b_1");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(myType) && "sizeof(T) == sizeof(myType)");
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_b_1->get64Bits();
	}

	if (r_type == "c_1")
	{	
		assert((partyNum == PARTY_A or partyNum == PARTY_C) && "Only A and C can call for c_1");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(myType) && "sizeof(T) == sizeof(myType)");
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_c_1->get64Bits();
	}

	if (r_type == "a_2")
	{
		assert((partyNum == PARTY_B or partyNum == PARTY_C) && "Only B and C can call for a_2");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(myType) && "sizeof(T) == sizeof(myType)");
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_a_2->get64Bits();
	}

	if (r_type == "b_2")
	{
		assert((partyNum == PARTY_B or partyNum == PARTY_C) && "Only B and C can call for b_2");
		assert(neg_type == "POSITIVE" && "neg_type should be POSITIVE");
		assert(sizeof(T) == sizeof(myType) && "sizeof(T) == sizeof(myType)");
		for (size_t i = 0; i < size; ++i)
			vec[i] = aes_b_2->get64Bits();
	}	
}


template<typename T>
void addVectors(const vector<T> &a, const vector<T> &b, vector<T> &c, size_t size)
{
	for (size_t i = 0; i < size; ++i)
		c[i] = a[i] + b[i];
}

template<typename T>
void subtractVectors(const vector<T> &a, const vector<T> &b, vector<T> &c, size_t size)
{
	for (size_t i = 0; i < size; ++i)
		c[i] = a[i] - b[i];
}

template<typename T>
void copyVectors(const vector<T> &a, vector<T> &b, size_t size)
{
	for (size_t i = 0; i < size; ++i)
		b[i] = a[i];
}


template<typename T1, typename T2>
void addModuloOdd(const vector<T1> &a, const vector<T2> &b, vector<myType> &c, size_t size)
{
	assert((sizeof(T1) == sizeof(myType) or sizeof(T2) == sizeof(myType)) && "At least one type should be myType for typecast to work");

	for (size_t i = 0; i < size; ++i)
	{
		if (a[i] == MINUS_ONE and b[i] == MINUS_ONE)
			c[i] = 0;
		else 
			c[i] = (a[i] + b[i] + wrapAround(a[i], b[i])) % MINUS_ONE;
	}
}

template<typename T1, typename T2>
void subtractModuloOdd(const vector<T1> &a, const vector<T2> &b, vector<myType> &c, size_t size)
{
	vector<myType> temp(size);
	for (size_t i = 0; i < size; ++i)
		temp[i] = MINUS_ONE - b[i];

	addModuloOdd<T1, myType>(a, temp, c, size);
}

template<typename T>
void sharesModuloOdd(vector<myType> &shares_1, vector<myType> &shares_2, 
				  const vector<T> &x, size_t size, string r_type)
{
	assert((r_type == "COMMON" or r_type == "INDEP") && "invalid randomness type for sharesOfBits");

	if (r_type == "COMMON")
	{
		for (size_t i = 0; i < size; ++i)
			shares_1[i] = aes_common->randModuloOdd();
	}

	if (r_type == "INDEP")
	{
		for (size_t i = 0; i < size; ++i)
			shares_1[i] = aes_indep->randModuloOdd();
	}

	subtractModuloOdd<T, myType>(x, shares_1, shares_2, size);
}

template<typename T1, typename T2>
myType addModuloOdd(T1 a, T2 b)
{
	assert((sizeof(T1) == sizeof(myType) or sizeof(T2) == sizeof(myType)) && "At least one type should be myType for typecast to work");

	if (a == MINUS_ONE and b == MINUS_ONE)
		return 0;
	else 
		return (a + b + wrapAround(a, b)) % MINUS_ONE;
}

template<typename T1, typename T2>
myType subtractModuloOdd(T1 a, T2 b)
{
	myType temp = MINUS_ONE - b;
	return addModuloOdd<T1, myType>(a, temp);
}

template<typename T>
void getVectorfromPrimary(vector<T> &vec, size_t size, string r_mode, string n_mode)
{
	assert(r_mode == "RANDOMIZE" or r_mode == "AS-IS" && "Random mode issue in getVectorfromPrimary");
	assert(n_mode == "NATURAL" or n_mode == "UNNATURAL" && "Natural mode issue in getVectorfromPrimary");

	if (r_mode == "RANDOMIZE")
	{
		//Original vec also gets modified here.
		vector<myType> temp(size);
		if (PRIMARY)
		{
			populateRandomVector<myType>(temp, size, "COMMON", "NEGATIVE");
			addVectors<myType>(vec, temp, vec, size);
		}
	}

	if (n_mode == "NATURAL")
	{
		if (PRIMARY)
			sendVector<T>(vec, partner(partyNum), size);
		if (!PRIMARY)
			receiveVector<T>(vec, partner(partyNum), size);
	}
	else
	{
		if (partyNum == PARTY_A)
			sendVector<T>(vec, PARTY_D, size);
		if (partyNum == PARTY_B)
			sendVector<T>(vec, PARTY_C, size);

		if (partyNum == PARTY_C)
			receiveVector<T>(vec, PARTY_B, size);
		if (partyNum == PARTY_D)
			receiveVector<T>(vec, PARTY_A, size);
	}
}

#endif

