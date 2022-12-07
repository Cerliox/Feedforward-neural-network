#pragma once
#ifndef _VECTOR_
#include <vector>
#endif

#ifdef _GNN_UTILS_
namespace Random {
	float RandomFloat(float, float);
	int RandomInt(int, int);
	double RandomDouble(double, double);
}

namespace ActivationFunction {
	float Linear(float);
	float ReLU(float);
	float Sigmoid(float);
	float Step(float);
}
#endif

struct Dim {
	unsigned int x = 0, y = 0;
};

struct NeuralNetwork {
	struct InputParameter {
		struct InputWeight {
			int a = 0, b = 0;
			int l = 0;
			float weight = 0.0f;
		};
		struct InputBias {
			int a;
			int l;
			float bias = 0.0f;
		};

		std::vector<float (*)(float)> activation_functions;
		unsigned int* layersizes;
		unsigned int amount_layers;

		std::vector<InputWeight> weights;
		std::vector<InputBias> biases;
	};
	struct Layer {
		unsigned int* layersizes;
		std::vector<float (*)(float)> activation_functions;
		unsigned int amount_layers;
	};
	struct Weightlayer {
		Dim* weightlayersizes;
		unsigned int amount_weightlayers;
	};

	Layer layer;
	Weightlayer weightlayer;
	float** weights;
	float** biases;

	float* output = nullptr;

	int error = -1;

	void Compile(InputParameter&);
	void Print();
	void PrintOutput();
	void Run(float*, unsigned int);
	void RandomWeights();
	void RandomBiases();

	void Multiply(float*&, unsigned int&, float*, Dim);
	void Add(float*, unsigned int, float*);
};

struct Gene {
	
};
struct GeneticNeuralNetwork {
	Gene* genes;
	unsigned int genes_size;
	
	bool compiled = false;
	NeuralNetwork net;

	unsigned int input_size = 1;
	unsigned int output_size = 1;
};