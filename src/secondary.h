
#ifndef SECONDARY_H
#define SECONDARY_H

#pragma once
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "secCompMultiParty.h"
#include "basicSockets.h"
#include <sstream>
#include "../util/TedKrovetzAesNiWrapperC.h"
#include "tools.h"
#include "globals.h"
#include <thread>
#include <iomanip>
#include <fstream>

void parseInputs(int argc, char* argv[]);
void initializeMPC();
void loadData(char* filename_train_data, char* filename_train_labels, 
			  char* filename_test_data, char* filename_test_labels);
void readMiniBatch(NeuralNetwork* net, string phase);
void train(NeuralNetwork* net, NeuralNetConfig* config);
void test(NeuralNetwork* net);

void deleteObjects();
#endif