
#pragma once
#include "LayerConfig.h"
#include "globals.h"
using namespace std;

class CNNConfig : public LayerConfig
{
public:
	size_t filters = 0;			//#Output feature maps
	size_t inputFeatures = 0;	//#Input feature maps
	size_t filterHeight = 0;
	size_t filterWidth = 0;

	size_t batchSize = 0;
	size_t imageHeight = 0;
	size_t imageWidth = 0;

	size_t poolSizeX = 0;
	size_t poolSizeY = 0;

	CNNConfig(size_t _filters, size_t _inputFeatures, size_t _filterHeight,
	size_t _filterWidth, size_t _batchSize, size_t _imageHeight,
	size_t _imageWidth, size_t _poolSizeX, size_t _poolSizeY)
	:filters(_filters),
	 inputFeatures(_inputFeatures),
	 filterHeight(_filterHeight),
	 filterWidth(_filterWidth),
	 batchSize(_batchSize),
	 imageHeight(_imageHeight),
	 imageWidth(_imageWidth),
	 poolSizeX(_poolSizeX),
	 poolSizeY(_poolSizeY),
	 LayerConfig("CNN")
	{
		assert((imageWidth - filterWidth + 1)%poolSizeX == 0 && "Check implementations for this condition");
		assert((imageHeight - filterHeight + 1)%poolSizeY == 0 && "Check implementations for this condition");
	};
};
