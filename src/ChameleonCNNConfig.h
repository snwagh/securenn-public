
#pragma once
#include "LayerConfig.h"
#include "globals.h"
using namespace std;

class ChameleonCNNConfig : public LayerConfig
{
public:
	size_t filters = 0;			//#Output feature maps
	size_t inputFeatures = 0;	//#Input feature maps
	size_t filterHeight = 0;
	size_t filterWidth = 0;

	size_t batchSize = 0;
	size_t imageHeight = 0;
	size_t imageWidth = 0;

	size_t strideX = 0;
	size_t strideY = 0;

	ChameleonCNNConfig(size_t _filters, size_t _inputFeatures, size_t _filterHeight,
	size_t _filterWidth, size_t _batchSize, size_t _imageHeight,
	size_t _imageWidth, size_t _strideX, size_t _strideY)
	:filters(_filters),
	 inputFeatures(_inputFeatures),
	 filterHeight(_filterHeight),
	 filterWidth(_filterWidth),
	 batchSize(_batchSize),
	 imageHeight(_imageHeight),
	 imageWidth(_imageWidth),
	 strideX(_strideX),
	 strideY(_strideY),
	 LayerConfig("ChameleonCNN")
	{
		assert((imageWidth)%strideX == 0 && "Check implementations for this condition");
		assert((imageHeight)%strideY == 0 && "Check implementations for this condition");
	};
};
