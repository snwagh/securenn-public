
#ifndef GLOBALS_H
#define GLOBALS_H

#pragma once
#include <emmintrin.h>
#include <vector>
#include <string>
#include <assert.h>
#include <limits.h>
using namespace std;

//Macros
#define _aligned_malloc(size,alignment) aligned_alloc(alignment,size)
#define _aligned_free free
#define getrandom(min, max) ((rand()%(int)(((max) + 1)-(min)))+ (min))
#define floatToMyType(a) ((myType)(a * (1 << FLOAT_PRECISION)))


//AES and other globals
#define LOG_DEBUG false
#define RANDOM_COMPUTE 256//Size of buffer for random elements
#define FIXED_KEY_AES "43739841701238781571456410093f43"
#define STRING_BUFFER_SIZE 256
#define true 1
#define false 0
#define DEBUG_CONST 16
#define DEBUG_INDEX 0
#define DEBUG_PRINT "SIGNED"
#define CPP_ASSEMBLY 1
#define MNIST false
#define PARALLEL true
#define NO_CORES 4


//MPC globals
extern int NUM_OF_PARTIES;
#define STANDALONE (NUM_OF_PARTIES == 1)
#define THREE_PC (NUM_OF_PARTIES == 3)
#define FOUR_PC (NUM_OF_PARTIES == 4)
#define PARTY_A 0
#define PARTY_B 1
#define PARTY_C 2
#define PARTY_D 3
#define PARTY_S 4

#define PRIME_NUMBER 67
#define FLOAT_PRECISION 13
#define PRIMARY (partyNum == PARTY_A or partyNum == PARTY_B)
#define	NON_PRIMARY (partyNum == PARTY_C or partyNum == PARTY_D)
#define HELPER (partyNum == PARTY_C)
#define MPC (FOUR_PC or THREE_PC)



//Neural Network globals.
//Batch size has to be a power of two
#define NUM_LAYERS 5
#define LL (NUM_LAYERS-2)
#define LAYER0 784
#define LAYER1 128
#define LAYER2 128
#define LAST_LAYER_SIZE 10
#if MNIST
	#define TRAINING_DATA_SIZE 60000
	#define TEST_DATA_SIZE 10000
	#define LOG_MINI_BATCH 7
#else
	#define TRAINING_DATA_SIZE 8
	#define TEST_DATA_SIZE 8
	#define LOG_MINI_BATCH 0
#endif
#define MINI_BATCH_SIZE (1 << LOG_MINI_BATCH)
#define LOG_LEARNING_RATE 5
#define LEARNING_RATE (1 << (FLOAT_PRECISION - LOG_LEARNING_RATE))
#define NO_OF_EPOCHS 1.5
#define NUM_ITERATIONS 10
// #define NUM_ITERATIONS ((int) (NO_OF_EPOCHS * TRAINING_DATA_SIZE/MINI_BATCH_SIZE))



//Typedefs and others
typedef __m128i superLongType;
typedef uint64_t myType;
typedef uint8_t smallType;

const int BIT_SIZE = (sizeof(myType) * CHAR_BIT);
const myType LARGEST_NEG = ((myType)1 << (BIT_SIZE - 1));
const myType MINUS_ONE = (myType)-1;
const smallType BOUNDARY = (256/PRIME_NUMBER) * PRIME_NUMBER;

const __m128i BIT1 = _mm_setr_epi8(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT2 = _mm_setr_epi8(2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT4 = _mm_setr_epi8(4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT8 = _mm_setr_epi8(8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT16 = _mm_setr_epi8(16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT32 = _mm_setr_epi8(32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT64 = _mm_setr_epi8(64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
const __m128i BIT128 = _mm_setr_epi8(128,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);


#endif