#pragma once
#include "CNNLayer.h"
using namespace std;


CNNLayer::CNNLayer(CNNConfig* conf)
:conf(conf->filters, conf->inputFeatures, conf->filterHeight,
	  conf->filterWidth, conf->batchSize, conf->imageHeight,
	  conf->imageWidth, conf->poolSizeX, conf->poolSizeY),
 weights(conf->filterHeight * conf->filterWidth * conf->inputFeatures * conf->filters),
 biases(conf->filters),
 activations(conf->batchSize * conf->filters * 
		    ((conf->imageWidth - conf->filterWidth + 1)/conf->poolSizeX) * 
 		    ((conf->imageHeight - conf->filterHeight + 1)/conf->poolSizeY)),
 deltas(conf->batchSize * conf->filters * 
       ((conf->imageWidth - conf->filterWidth + 1)/conf->poolSizeX) * 
       ((conf->imageHeight - conf->filterHeight + 1)/conf->poolSizeY)),
 reluPrimeSmall(conf->batchSize * conf->filters *  
		    ((conf->imageWidth - conf->filterWidth + 1)/conf->poolSizeX) * 
 		    ((conf->imageHeight - conf->filterHeight + 1)/conf->poolSizeY)),
 reluPrimeLarge(conf->batchSize * conf->filters *  
		    ((conf->imageWidth - conf->filterWidth + 1)/conf->poolSizeX) * 
 		    ((conf->imageHeight - conf->filterHeight + 1)/conf->poolSizeY)),
 maxIndex(conf->batchSize * conf->filters *  
		    ((conf->imageWidth - conf->filterWidth + 1)/conf->poolSizeX) * 
 		    ((conf->imageHeight - conf->filterHeight + 1)/conf->poolSizeY)),
 deltaRelu(conf->batchSize * conf->filters *  
		    ((conf->imageWidth - conf->filterWidth + 1)/conf->poolSizeX) * 
 		    ((conf->imageHeight - conf->filterHeight + 1)/conf->poolSizeY))
{
	initialize();
};


void CNNLayer::initialize()
{
	//Initialize weights and biases here.
	//Ensure that initialization is correctly done.
	size_t lower = 30;
	size_t higher = 50;
	size_t decimation = 10000;
	size_t size = weights.size();

	vector<myType> temp(size);
	for (size_t i = 0; i < size; ++i)
		temp[i] = floatToMyType(1);
		// temp[i] = floatToMyType((float)(rand() % (higher - lower) + lower)/decimation);

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


void CNNLayer::forward(const vector<myType>& inputActivation)
{
	log_print("CNN.forward");

	size_t B = conf.batchSize;
	size_t iw = conf.imageWidth;
	size_t ih = conf.imageHeight;
	size_t fw = conf.filterWidth;
	size_t fh = conf.filterHeight;
	size_t D = conf.filters;
	size_t C = conf.inputFeatures;
	size_t px = conf.poolSizeX;
	size_t py = conf.poolSizeY;
	size_t p_range = (ih-fh+1);
	size_t q_range = (iw-fw+1);
	size_t alpha_range = (iw-fw+1)/px;
	size_t beta_range = (ih-fh+1)/py;

	//Reshape weights
	size_t size_rw = fw*fh*C*D;
	size_t rows_rw = fw*fh*C;
	size_t columns_rw = D;
	vector<myType> reshapedWeights(size_rw, 0);
	for (int i = 0; i < size_rw; ++i)
		reshapedWeights[(i%rows_rw)*columns_rw + (i/rows_rw)] = weights[i];

	//reshape activations
	size_t size_convo = (p_range*q_range*B) * (fw*fh*C); 
	vector<myType> convShaped(size_convo, 0);
	convolutionReshape(inputActivation, convShaped, iw, ih, C, B, fw, fh, 1, 1);


	//Convolution multiplication
	vector<myType> convOutput(p_range*q_range*B*D, 0);
	if (STANDALONE)
	{
		matrixMultEigen(convShaped, reshapedWeights, convOutput, 
					(p_range*q_range*B), (fw*fh*C), D, 0, 0);
		dividePlainSA(convOutput, (1 << FLOAT_PRECISION));
	}

	if (MPC)
	{
		funcMatMulMPC(convShaped, reshapedWeights, convOutput, 
					(p_range*q_range*B), (fw*fh*C), D, 0, 0);
	}

	//Add Biases
	if (PRIMARY or STANDALONE)
	{
		size_t rows_convo = p_range*q_range*B;
		size_t columns_convo = D;
		for(size_t r = 0; r < rows_convo; ++r)
			for(size_t c = 0; c < columns_convo; ++c)
				convOutput[r*columns_convo + c] += biases[c];
	}

	//reshape convOutput into x
	size_t size_x = p_range*q_range*D*B;
	size_t size_image = p_range*q_range;
	size_t size_batch = p_range*q_range*D;
	vector<myType> x(size_x, 0);
	if (PRIMARY or STANDALONE)
		for (size_t i = 0; i < size_x; ++i)
			x[(i/size_batch)*size_batch + (i%D)*size_image + ((i/D)%size_image)] = convOutput[i];


	//Relu
	vector<myType> y(size_x/(px*py), 0);
	vector<myType> maxPoolShaped(size_x, 0);
	maxPoolReshape(x, maxPoolShaped, ih-fh+1, iw-fw+1, D, B, py, px, py, px);
	findMax(maxPoolShaped, y, maxIndex, size_x/(px*py), px*py);


	if (STANDALONE)
	{
		for (size_t index = 0; index < size_x/(px*py); ++index)
		{
			reluPrimeSmall[index] = (y[index] < LARGEST_NEG ? 1:0);
			activations[index] = reluPrimeSmall[index]*y[index];
		}
	}

	if (THREE_PC)
	{
		funcRELUPrime3PC(y, reluPrimeLarge, size_x/(px*py));
		funcSelectShares3PC(y, reluPrimeLarge, activations, size_x/(px*py));
	}

	if (FOUR_PC)
	{
		funcRELUPrime4PC(y, reluPrimeSmall, size_x/(px*py));
		funcSelectShares4PC(y, reluPrimeSmall, activations, size_x/(px*py));
	}

	// vector<myType> maxPoolShaped(size_x, 0);
	// maxPoolReshape(y, maxPoolShaped, ih-fh+1, iw-fw+1, D, B, py, px, py, px);
	// findMax(maxPoolShaped, activations, maxIndex, size_x/(px*py), px*py);

	// cout << "MaxPool Output Size: " << activations.size() << " = batchSize x " 
	// 					  << activations.size()/MINI_BATCH_SIZE << endl;

	// if (MPC)
	// 	if (PRIMARY)
	// 		funcReconstruct2PC(activations, DEBUG_CONST, "activations");
}


void CNNLayer::computeDelta(vector<myType>& prevDelta)
{
	log_print("CNN.computeDelta");

	if (STANDALONE)
		computeDeltaSA(prevDelta);
	else
		computeDeltaMPC(prevDelta);	
}

void CNNLayer::updateEquations(const vector<myType>& prevActivations)
{
	log_print("CNN.updateEquations");

	if (STANDALONE)
		updateEquationsSA(prevActivations);
	else
		updateEquationsMPC(prevActivations);
}

void CNNLayer::findMax(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
						size_t rows, size_t columns)
{
	log_print("CNN.findMax");

	if (STANDALONE)
		maxSA(a, max, maxIndex, rows, columns);
	else
		maxMPC(a, max, maxIndex, rows, columns);
}






/******************************** Standalone ********************************/
void CNNLayer::computeDeltaSA(vector<myType>& prevDelta)
{
	size_t B = conf.batchSize;
	size_t iw = conf.imageWidth;
	size_t ih = conf.imageHeight;
	size_t fw = conf.filterWidth;
	size_t fh = conf.filterHeight;
	size_t D = conf.filters;
	size_t C = conf.inputFeatures;
	size_t px = conf.poolSizeX;
	size_t py = conf.poolSizeY;
	size_t p_range = (ih-fh+1);
	size_t q_range = (iw-fw+1);
	size_t alpha_range = (ih-fh+1)/py;
	size_t beta_range = (iw-fw+1)/px;
	size_t size_y = (p_range*q_range*D*B);
	size_t size_delta = alpha_range*beta_range*D*B;
	vector<myType> deltaMaxPool(size_y);


	//Dot product with relu'
	for (size_t i = 0; i < size_delta; ++i)
		deltaRelu[i] = deltas[i] * reluPrimeSmall[i];	


	//Populate thatMatrix
	vector<myType> thatMatrixTemp(size_y, 0), thatMatrix(size_y, 0);
	for (size_t i = 0; i < size_delta; ++i)
		thatMatrixTemp[i*px*py + maxIndex[i]] = 1;


	//Reshape thatMatrix
	size_t repeat_size = D*B;
	size_t alpha_offset, beta_offset, alpha, beta;
	for (size_t r = 0; r < repeat_size; ++r)
	{
		size_t size_temp = p_range*q_range;
		for (size_t i = 0; i < size_temp; ++i)
		{
			alpha = (i/(px*py*beta_range));
			beta = (i/(px*py)) % beta_range;
			alpha_offset = (i%(px*py))/px;
			beta_offset = (i%py);
			thatMatrix[((py*alpha + alpha_offset)*q_range) + 
					   (px*beta + beta_offset) + r*size_temp] 
			= thatMatrixTemp[r*size_temp + i];
		}
	}

	//Replicate delta martix appropriately
	vector<myType> largerDelta(size_y, 0);
	size_t index_larger, index_smaller;
	for (size_t r = 0; r < repeat_size; ++r)
	{
		size_t size_temp = p_range*q_range;
		for (size_t i = 0; i < size_temp; ++i)
		{
			index_smaller = r*size_temp/(px*py) + (i/(q_range*py))*beta_range + ((i%q_range)/px);
			index_larger = r*size_temp + (i/q_range)*q_range + (i%q_range);
			largerDelta[index_larger] = deltaRelu[index_smaller];
		}
	}

	//Dot product
	for (size_t i = 0; i < size_y; ++i)
		deltaMaxPool[i] = largerDelta[i] * thatMatrix[i]; 


	//Final stage of delta back-prop
	//reverse and reshape weights.
	size_t size_w = fw*fh*C*D;
	size_t size_D = C*fw*fh;
	size_t size_C = fw*fh;
	vector<myType> reshapedWeights(size_w, 0);
	for (size_t i = 0; i < size_w; ++i)
		reshapedWeights[((i/size_D)*size_D) + ((i%size_C)*C) + ((i%size_D)/size_C)] 
		= weights[(i/size_C)*size_C + size_C - 1 - (i%size_C)];


	//ZeroPad delta.
	size_t x_range = (iw+fw-1);
	size_t y_range = (ih+fh-1);
	size_t size_zeroPad = x_range*y_range*D*B;
	vector<myType> zeroPaddedDelta(size_zeroPad, 0);
	repeat_size = D*B;
	for (size_t r = 0; r < repeat_size; ++r)
	{
		for (size_t p = 0; p < p_range; ++p)
		{
			for (size_t q = 0; q < q_range; ++q)
			{
				index_smaller = r*(p_range*q_range) + (p*q_range) + q;
				index_larger = r*(x_range*y_range) + (p+fh-1)*x_range + (q+fw-1);
				zeroPaddedDelta[index_larger] = deltaRelu[index_smaller];
			}
		}
	}


	//convReshape delta matrix
	vector<myType> reshapedPaddedDelta((iw*ih*B) * (fw*fh*D), 0);
	convolutionReshape(zeroPaddedDelta, reshapedPaddedDelta, 
							   iw+fw-1, ih+fh-1, D, B, fh, fw, 1, 1);


	//Mat-Mul
	size_t size_pd = iw*ih*C*B;
	size_t size_batch = iw*ih*C;
	size_C = iw*ih;
	vector<myType> temp(size_pd, 0); 
	matrixMultEigen(reshapedPaddedDelta, reshapedWeights, temp, 
				   (iw*ih*B), (fw*fh*D), C, 0, 0);
	dividePlainSA(prevDelta, (1 << FLOAT_PRECISION));


	//Reshape temp into prevDelta
	for (size_t i = 0; i < size_pd; ++i)
		prevDelta[((i/size_batch)*size_batch) + ((i%C)*size_C) + ((i%size_batch)/C)] = temp[i];
}


void CNNLayer::updateEquationsSA(const vector<myType>& prevActivations)
{
	size_t B = conf.batchSize;
	size_t iw = conf.imageWidth;
	size_t ih = conf.imageHeight;
	size_t fw = conf.filterWidth;
	size_t fh = conf.filterHeight;
	size_t D = conf.filters;
	size_t C = conf.inputFeatures;
	size_t p_range = (ih-fh+1);
	size_t q_range = (iw-fw+1);
	size_t px = conf.poolSizeX;
	size_t py = conf.poolSizeY;
	size_t alpha_range = (ih-fh+1)/px;
	size_t beta_range = (iw-fw+1)/py;

	//Update bias
	vector<myType> temp(D, 0);
	size_t size_batch = D*p_range*q_range;
	size_t size_D = p_range*q_range;
	size_t loc = 0;
	for (size_t i = 0; i < D; ++i)
		for (size_t j = 0; j < B; ++j)
			for (size_t k = 0; k < p_range; k++)
				for (size_t l = 0; l < q_range; l++)
					{
						loc = j*size_batch + i*size_D + k*q_range + l;
						temp[i] += deltaRelu[loc];
					}

	for (size_t i = 0; i < D; ++i)
		biases[i] -= dividePlainSA(multiplyMyTypesSA(temp[i], LEARNING_RATE, FLOAT_PRECISION), B);


	//Update Weights
	vector<myType> shapedDelta(B*D*p_range*q_range);
	size_batch = D*p_range*q_range;
	size_D = p_range*q_range;
	size_t counter = 0;
	for (size_t i = 0; i < B; ++i)
		for (size_t j = 0; j < p_range; j++)
			for (size_t k = 0; k < q_range; k++)
				for (size_t l = 0; l < D; ++l)
				{
					loc = i*size_batch + l*size_D + j*q_range + k;
					shapedDelta[counter++] = deltaRelu[loc];
				}


	vector<myType> shapedActivation(B*C*p_range*q_range*fw*fh);
	size_batch = C*iw*ih;
	size_t size_C = iw*ih;
	counter = 0;
	for (size_t i = 0; i < C; ++i)
		for (size_t j = 0; j < fh; j++)
			for (size_t k = 0; k < fw; k++)
				for (size_t l = 0; l < B; l++)
				{
					loc = l*size_batch + i*size_C + j*iw+ k;
					for (size_t a = 0; a < q_range; ++a)
						for (size_t b = 0; b < p_range; ++b)
							shapedActivation[counter++] = prevActivations[loc + a*iw + b];
				}



	size_t size_w = fw*fh*C*D;
	vector<myType> tempProd(size_w, 0);
	matrixMultEigen(shapedActivation, shapedDelta, tempProd, 
				   (C*fw*fh), (p_range*q_range*B), D, 0, 0);
	dividePlainSA(tempProd, (1 << FLOAT_PRECISION));

	//Reorganize weight gradient
	vector<myType> tempShaped(size_w, 0);
	size_t rows_ts = (fw*fh*C);
	size_t columns_ts = D;
	for (size_t i = 0; i < rows_ts; ++i)
		for (size_t j = 0; j < columns_ts; j++)
			tempShaped[j*rows_ts + i] = tempProd[i*columns_ts + j];


	for (size_t i = 0; i < size_w; ++i)
		weights[i] -= dividePlainSA(multiplyMyTypesSA(tempShaped[i], LEARNING_RATE, FLOAT_PRECISION), B);
}


void CNNLayer::maxSA(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
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
void CNNLayer::computeDeltaMPC(vector<myType>& prevDelta)
{
	//For 4PC ensure delta sharing
	size_t B = conf.batchSize;
	size_t iw = conf.imageWidth;
	size_t ih = conf.imageHeight;
	size_t fw = conf.filterWidth;
	size_t fh = conf.filterHeight;
	size_t D = conf.filters;
	size_t C = conf.inputFeatures;
	size_t px = conf.poolSizeX;
	size_t py = conf.poolSizeY;
	size_t p_range = (ih-fh+1);
	size_t q_range = (iw-fw+1);
	size_t alpha_range = (ih-fh+1)/py;
	size_t beta_range = (iw-fw+1)/px;
	size_t size_y = (p_range*q_range*D*B);
	size_t size_delta = alpha_range*beta_range*D*B;
	vector<myType> deltaMaxPool(size_y);


	//Dot product with relu'
	if (THREE_PC)
		funcDotProductMPC(deltaRelu, reluPrimeLarge, deltas, size_delta);

	if (FOUR_PC)
	{
		vector<myType> temp(size_delta);
		for (size_t i = 0; i < size_delta; ++i)
			temp[i] = reluPrimeSmall[i];

		funcDotProductMPC(deltaRelu, temp, deltas, size_delta);
	}



	//Populate thatMatrix
	vector<myType> thatMatrixTemp(size_y, 0), thatMatrix(size_y, 0);
	funcMaxIndexMPC(thatMatrixTemp, maxIndex, size_delta, px*py);

	//Reshape thatMatrix
	size_t repeat_size = D*B;
	size_t alpha_offset, beta_offset, alpha, beta;
	for (size_t r = 0; r < repeat_size; ++r)
	{
		size_t size_temp = p_range*q_range;
		for (size_t i = 0; i < size_temp; ++i)
		{
			alpha = (i/(px*py*beta_range));
			beta = (i/(px*py)) % beta_range;
			alpha_offset = (i%(px*py))/px;
			beta_offset = (i%py);
			thatMatrix[((py*alpha + alpha_offset)*q_range) + 
					   (px*beta + beta_offset) + r*size_temp] 
			= thatMatrixTemp[r*size_temp + i];
		}
	}

	//Replicate delta martix appropriately
	vector<myType> largerDelta(size_y, 0);
	size_t index_larger, index_smaller;
	for (size_t r = 0; r < repeat_size; ++r)
	{
		size_t size_temp = p_range*q_range;
		for (size_t i = 0; i < size_temp; ++i)
		{
			index_smaller = r*size_temp/(px*py) + (i/(q_range*py))*beta_range + ((i%q_range)/px);
			index_larger = r*size_temp + (i/q_range)*q_range + (i%q_range);
			largerDelta[index_larger] = deltaRelu[index_smaller];
		}
	}

	//Dot product
	funcDotProductMPC(largerDelta, thatMatrix, deltaMaxPool, size_y);




	//Final stage of delta back-prop
	//reverse and reshape weights.
	size_t size_w = fw*fh*C*D;
	size_t size_D = C*fw*fh;
	size_t size_C = fw*fh;
	vector<myType> reshapedWeights(size_w, 0);
	for (size_t i = 0; i < size_w; ++i)
		reshapedWeights[((i/size_D)*size_D) + ((i%size_C)*C) + ((i%size_D)/size_C)] 
		= weights[(i/size_C)*size_C + size_C - 1 - (i%size_C)];


	//ZeroPad delta.
	size_t x_range = (iw+fw-1);
	size_t y_range = (ih+fh-1);
	size_t size_zeroPad = x_range*y_range*D*B;
	vector<myType> zeroPaddedDelta(size_zeroPad, 0);
	repeat_size = D*B;
	for (size_t r = 0; r < repeat_size; ++r)
	{
		for (size_t p = 0; p < p_range; ++p)
		{
			for (size_t q = 0; q < q_range; ++q)
			{
				index_smaller = r*(p_range*q_range) + (p*q_range) + q;
				index_larger = r*(x_range*y_range) + (p+fh-1)*x_range + (q+fw-1);
				zeroPaddedDelta[index_larger] = deltaRelu[index_smaller];
			}
		}
	}


	//convReshape delta matrix
	vector<myType> reshapedPaddedDelta((iw*ih*B) * (fw*fh*D), 0);
	convolutionReshape(zeroPaddedDelta, reshapedPaddedDelta, 
							   iw+fw-1, ih+fh-1, D, B, fh, fw, 1, 1);


	//Mat-Mul
	size_t size_pd = iw*ih*C*B;
	size_t size_batch = iw*ih*C;
	size_C = iw*ih;
	vector<myType> temp(size_pd, 0); 

	funcMatMulMPC(reshapedPaddedDelta, reshapedWeights, prevDelta, 
					(iw*ih*B), (fw*fh*D), C, 0, 0);


	//Reshape temp into prevDelta
	for (size_t i = 0; i < size_pd; ++i)
		prevDelta[((i/size_batch)*size_batch) + ((i%C)*size_C) + ((i%size_batch)/C)] = temp[i];
}

void CNNLayer::updateEquationsMPC(const vector<myType>& prevActivations)
{
	size_t B = conf.batchSize;
	size_t iw = conf.imageWidth;
	size_t ih = conf.imageHeight;
	size_t fw = conf.filterWidth;
	size_t fh = conf.filterHeight;
	size_t D = conf.filters;
	size_t C = conf.inputFeatures;
	size_t p_range = (ih-fh+1);
	size_t q_range = (iw-fw+1);
	size_t px = conf.poolSizeX;
	size_t py = conf.poolSizeY;
	size_t alpha_range = (ih-fh+1)/px;
	size_t beta_range = (iw-fw+1)/py;

	//Update bias
	vector<myType> temp(D, 0);
	size_t size_batch = D*p_range*q_range;
	size_t size_D = p_range*q_range;
	size_t loc = 0;
	for (size_t i = 0; i < D; ++i)
		for (size_t j = 0; j < B; ++j)
			for (size_t k = 0; k < p_range; k++)
				for (size_t l = 0; l < q_range; l++)
					{
						loc = j*size_batch + i*size_D + k*q_range + l;
						temp[i] += deltaRelu[loc];
					}

	if (PRIMARY)
		funcTruncate2PC(temp, LOG_MINI_BATCH + LOG_LEARNING_RATE, D, PARTY_A, PARTY_B);

	subtractVectors<myType>(biases, temp, biases, D);



	//Update Weights
	vector<myType> shapedDelta(B*D*p_range*q_range);
	size_batch = D*p_range*q_range;
	size_D = p_range*q_range;
	size_t counter = 0;
	for (size_t i = 0; i < B; ++i)
		for (size_t j = 0; j < p_range; j++)
			for (size_t k = 0; k < q_range; k++)
				for (size_t l = 0; l < D; ++l)
				{
					loc = i*size_batch + l*size_D + j*q_range + k;
					shapedDelta[counter++] = deltaRelu[loc];
				}


	vector<myType> shapedActivation(B*C*p_range*q_range*fw*fh);
	size_batch = C*iw*ih;
	size_t size_C = iw*ih;
	counter = 0;
	for (size_t i = 0; i < C; ++i)
		for (size_t j = 0; j < fh; j++)
			for (size_t k = 0; k < fw; k++)
				for (size_t l = 0; l < B; l++)
				{
					loc = l*size_batch + i*size_C + j*iw+ k;
					for (size_t a = 0; a < q_range; ++a)
						for (size_t b = 0; b < p_range; ++b)
							shapedActivation[counter++] = prevActivations[loc + a*iw + b];
				}



	size_t size_w = fw*fh*C*D;
	vector<myType> tempProd(size_w, 0);

	funcMatMulMPC(shapedActivation, shapedDelta, tempProd, 
				  (C*fw*fh), (p_range*q_range*B), D, 0, 0);

	if (PRIMARY)
		funcTruncate2PC(tempProd, LOG_MINI_BATCH + LOG_LEARNING_RATE, (C*fw*fh*D), PARTY_A, PARTY_B);

	//Reorganize weight gradient
	vector<myType> tempShaped(size_w, 0);
	size_t rows_ts = (fw*fh*C);
	size_t columns_ts = D;
	for (size_t i = 0; i < rows_ts; ++i)
		for (size_t j = 0; j < columns_ts; j++)
			tempShaped[j*rows_ts + i] = tempProd[i*columns_ts + j];


	subtractVectors<myType>(weights, tempShaped, weights, fw*fh*C*D);
}


void CNNLayer::maxMPC(vector<myType> &a, vector<myType> &max, vector<myType> &maxIndex, 
							size_t rows, size_t columns)
{
	funcMaxMPC(a, max, maxIndex, rows, columns);
}

