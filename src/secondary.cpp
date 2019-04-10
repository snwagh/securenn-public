
#include "secondary.h"
#include <iostream>
using namespace std;


//this player number
int partyNum;
//aes_key of the party
char *party_aes_key;


//For faster DGK computation
smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];


//communication
extern string * addrs;
extern BmrNet ** communicationSenders;
extern BmrNet ** communicationReceivers;


vector<myType> trainData, testData;
vector<myType> trainLabels, testLabels;
size_t trainDataBatchCounter = 0;
size_t trainLabelsBatchCounter = 0;
size_t testDataBatchCounter = 0;
size_t testLabelsBatchCounter = 0;




void parseInputs(int argc, char* argv[])
{	
	//assert(NUM_LAYERS == 4 && "Change LAST_LAYER_SIZE appropriately if using a different network");
	//If this fails, change functions in tools (divide and multiply ones)
	assert((sizeof(double) == sizeof(myType)) && "sizeof(double) != sizeof(myType)");

	if (argc < 10) 
		print_usage(argv[0]);

	if (strcmp(argv[1], "STANDALONE") == 0)
		NUM_OF_PARTIES = 1;
	else if (strcmp(argv[1], "3PC") == 0)
		NUM_OF_PARTIES = 3;
	else if (strcmp(argv[1], "4PC") == 0)
		NUM_OF_PARTIES = 4;

	partyNum = atoi(argv[2]);
	
	if (partyNum < 0 or partyNum > 4) 
		print_usage(argv[0]);

	loadData(argv[6], argv[7], argv[8], argv[9]);
}


void initializeMPC()
{
	//populate offline module prime addition and multiplication tables
	for (int i = 0; i < PRIME_NUMBER; ++i)
		for (int j = 0; j < PRIME_NUMBER; ++j)
		{
			additionModPrime[i][j] = (i + j) % PRIME_NUMBER;
			multiplicationModPrime[i][j] = (i * j) % PRIME_NUMBER;
		}
}



void loadData(char* filename_train_data, char* filename_train_labels, 
			  char* filename_test_data, char* filename_test_labels)
{
	float temp;
	ifstream f(filename_train_data);
	for (int i = 0; i < TRAINING_DATA_SIZE * LAYER0; ++i)
	{
		f >> temp;
		trainData.push_back(floatToMyType(temp));
	}
	f.close();

	ifstream g(filename_train_labels);
	for (int i = 0; i < TRAINING_DATA_SIZE * LAST_LAYER_SIZE; ++i)
	{
		g >> temp;
		trainLabels.push_back(floatToMyType(temp));
	}
	g.close();

	ifstream h(filename_test_data);
	for (int i = 0; i < TEST_DATA_SIZE * LAYER0; ++i)
	{
		h >> temp;
		testData.push_back(floatToMyType(temp));
	}
	h.close();

	ifstream k(filename_test_labels);
	for (int i = 0; i < TEST_DATA_SIZE * LAST_LAYER_SIZE; ++i)
	{
		k >> temp;
		testLabels.push_back(floatToMyType(temp));
	}
	k.close();	
}


void readMiniBatch(NeuralNetwork* net, string phase)
{
	size_t s = trainData.size();
	size_t t = trainLabels.size();

	if (phase == "TRAINING")
	{
		for (int i = 0; i < LAYER0 * MINI_BATCH_SIZE; ++i)
			net->inputData[i] = trainData[(trainDataBatchCounter + i)%s];

		for (int i = 0; i < LAST_LAYER_SIZE * MINI_BATCH_SIZE; ++i)
			net->outputData[i] = trainLabels[(trainLabelsBatchCounter + i)%t];

		trainDataBatchCounter += LAYER0 * MINI_BATCH_SIZE;
		trainLabelsBatchCounter += LAST_LAYER_SIZE * MINI_BATCH_SIZE;
	}

	if (trainDataBatchCounter > s)
		trainDataBatchCounter -= s;

	if (trainLabelsBatchCounter > t)
		trainLabelsBatchCounter -= t;



	size_t p = testData.size();
	size_t q = testLabels.size();

	if (phase == "TESTING")
	{
		for (int i = 0; i < LAYER0 * MINI_BATCH_SIZE; ++i)
			net->inputData[i] = testData[(testDataBatchCounter + i)%p];

		for (int i = 0; i < LAST_LAYER_SIZE * MINI_BATCH_SIZE; ++i)
			net->outputData[i] = testLabels[(testLabelsBatchCounter + i)%q];

		testDataBatchCounter += LAYER0 * MINI_BATCH_SIZE;
		testLabelsBatchCounter += LAST_LAYER_SIZE * MINI_BATCH_SIZE;
	}

	if (testDataBatchCounter > p)
		testDataBatchCounter -= p;

	if (testLabelsBatchCounter > q)
		testLabelsBatchCounter -= q;
}


void train(NeuralNetwork* net, NeuralNetConfig* config)
{
	log_print("train");

	if (!STANDALONE)
		initializeMPC();	

	for (int i = 0; i < config->numIterations; ++i)
	{
		// cout << "----------------------------------" << endl;  
		// cout << "Iteration " << i << endl;
		
		readMiniBatch(net, "TRAINING");

		// start_m();
		net->forward();
		// end_m("Forward");

		// start_m();
		net->backward();
		// end_m("Backward");

		// cout << "----------------------------------" << endl;  
	}

	// if (STANDALONE)
	// 	net->layers[LL].layerPrint(DEBUG_CONST, "WEIGHTS", DEBUG_PRINT);
	// else
	// 	if (PRIMARY)
	// 		net->layers[LL].funcReconstruct2PC(net->layers[LL].weights, DEBUG_CONST, "weights");
}


void test(NeuralNetwork* net)
{
	log_print("test");

	// counter[0]: Correct samples, counter[1]: total samples
	vector<size_t> counter(2,0);
	vector<myType> maxIndex(MINI_BATCH_SIZE);

	// for (int i = 0; i < TEST_DATA_SIZE/MINI_BATCH_SIZE; ++i)
	for (int i = 0; i < NUM_ITERATIONS; ++i)
	{
		readMiniBatch(net, "TESTING");
		
		// start_m();
		net->forward();
		// end_m("Forward");

		// start_m();
		net->predict(maxIndex);
		// end_m("prediction");

		// net->getAccuracy(maxIndex, counter);
	}
}



void deleteObjects()
{
	//close connection
	for (int i = 0; i < NUM_OF_PARTIES; i++)
	{
		if (i != partyNum)
		{
			delete communicationReceivers[i];
			delete communicationSenders[i];
		}
	}
	delete[] communicationReceivers;
	delete[] communicationSenders;

	delete[] addrs;
	delete[] party_aes_key;
}

