#define _GNN_UTILS_
#include "NeuralNetworks.h"

#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include "Clock.h"

#ifdef __AVX__

#define _USE_AVX 1

#if _USE_AVX == 1
#include <immintrin.h>

#define AVX_BSIZE 8 * 32
#define AVX_FWORDSIZE 8
#define _MM256_DP_PS_MASK 0b11111111
#endif
#endif

#ifdef _GNN_UTILS_
float ActivationFunction::Linear(float x) {
	return x;
}
float ActivationFunction::ReLU(float x) {
	return fmax(0.0, x);
}
float ActivationFunction::Sigmoid(float x) {
	return 1.0f / (1.0f + pow(M_E, -x));
}
float ActivationFunction::Step(float x) {
	return x >= 1.0 ? 1.0f : 0.0f;
}
float Random::RandomFloat(float min, float max) {
	float r = (float)rand() / (float)RAND_MAX;
	return min + r * (max - min);
}
double Random::RandomDouble(double min, double max) {
	double r = (double)rand() / (double)RAND_MAX;
	return min + r * (max - min);
}
int Random::RandomInt(int min, int max)
{
	if (max - min == 0)
		return 0;
	return min + rand() % (max - min);
}
#endif

void NeuralNetwork::Compile(InputParameter& ip) {
	layer.amount_layers = ip.amount_layers;
	layer.layersizes = ip.layersizes;
#if _USE_AVX == 1
	for (int i = 0; i < ip.amount_layers; i++) {
		if (ip.layersizes[i] % AVX_FWORDSIZE != 0) {
			ip.layersizes[i] += AVX_FWORDSIZE - (ip.layersizes[i] % AVX_FWORDSIZE);
		}
	}
#endif
	layer.activation_functions = ip.activation_functions;

	weightlayer.amount_weightlayers = layer.amount_layers - 1;
	weightlayer.weightlayersizes = new Dim[weightlayer.amount_weightlayers];
	weights = new float* [weightlayer.amount_weightlayers];
	biases = new float* [layer.amount_layers];

	for (int i = 0; i < ip.amount_layers-1; i++) {
		unsigned int am_input_neurons = ip.layersizes[i];
		unsigned int am_output_neurons = ip.layersizes[i + 1];
		unsigned int am_weights = am_input_neurons * am_output_neurons;

		weightlayer.weightlayersizes[i] = { am_output_neurons, am_input_neurons };

		weights[i] = new float[am_weights];
		memset(weights[i], 0x00, sizeof(float) * am_weights);
	}
	for (int i = 0; i < ip.amount_layers; i++) {
		biases[i] = new float[layer.layersizes[i]];

		memset(biases[i], 0x00, sizeof(float) * layer.layersizes[i]);
	}

	for (int i = 0; i < ip.weights.size(); i++) {
		InputParameter::InputWeight* iw = &ip.weights[i];
		if (iw->l >= weightlayer.amount_weightlayers) {
			error = 2;
			continue;
		}
		if (iw->a >= layer.layersizes[iw->l] || iw->b >= layer.layersizes[iw->l + 1]) {
			error = 3;
			continue;
		}

		weights[iw->l][iw->b * layer.layersizes[iw->l] + iw->a] = iw->weight;
	}
	for (int i = 0; i < ip.biases.size(); i++) {
		InputParameter::InputBias* ib;
		if (ib->l >= layer.amount_layers) {
			error = 4;
			continue;
		}
		if (ib->a >= layer.layersizes[ib->l]) {
			error = 5;
			continue;
		}

		biases[ib->l][ib->a] = ib->bias;

	}
	error = 0;
}
void NeuralNetwork::Print() {
	printf("Error: %i\n", error);

	std::cout << "Biases:" << std::endl;

	for (int i = 0; i < layer.amount_layers; i++) {
		printf("%i ", i);
		for (int a = 0; a < layer.layersizes[i]; a++) {
			printf(" %.2f ", biases[i][a]);
		}
		printf("\n");
	}

	std::cout << "Weights: " << std::endl;
	for (int i = 0; i < weightlayer.amount_weightlayers; i++) {
		printf("%i ", i);
		for (int a = 0; a < weightlayer.weightlayersizes[i].x; a++) {
			for (int b = 0; b < weightlayer.weightlayersizes[i].y; b++) {
				printf(" %.2f ", weights[i][a * layer.layersizes[i] + b]);
			}
			printf("\t");
		}
		printf("\n");
	}

	if (output != nullptr) {
		std::cout << "Output: " << std::endl;
		for (int i = 0; i < layer.layersizes[layer.amount_layers - 1]; i++) {
			printf("%.2f ", output[i]);
		}
		printf("\n");
	}
}
void NeuralNetwork::Run(float* input, unsigned int input_size) {
	if (input_size != layer.layersizes[0])
		return;

	if (output != nullptr) 
		delete[] output;
	
	output = new float[input_size];
	memcpy(output, input, sizeof(float) * input_size);
	unsigned int output_dim = input_size;

	for (int i = 0; i < weightlayer.amount_weightlayers; i++) {
		Multiply(output, output_dim, weights[i], weightlayer.weightlayersizes[i]);
		Add(output, output_dim, biases[i+1]);
		if (i < layer.activation_functions.size()) {
			for (int o = 0; o < output_dim; o++) {
				output[o] = layer.activation_functions[i](output[o]);
			}
		}
	}
}
void NeuralNetwork::Multiply(float*& in, unsigned int& dim, float* weights, Dim dweights) {
	unsigned int output_dim = dweights.x;
	
	float* output = new float[output_dim];
	memset(output, 0x00, sizeof(float) * output_dim);

#if _USE_AVX == 1

	alignas(32) float temp[AVX_FWORDSIZE];
	for (int o = 0; o < output_dim; o++) {
		for (int i = 0; i < dim; i += AVX_FWORDSIZE) {
			__m256 avx_in = _mm256_loadu_ps(in + i);
			__m256 avx_weight = _mm256_loadu_ps(weights + o * dweights.y + i);
			__m256 dst = _mm256_dp_ps(avx_in, avx_weight, _MM256_DP_PS_MASK);
			
			_mm256_store_ps(temp, dst);
			output[o] += temp[0];
		}
	}
#else

	for (int o = 0; o < output_dim; o++) {
		float val = 0.0f;
		for (int i = 0; i < dim; i++) {
			output[o] += weights[o * dweights.y + i] * in[i];
		}
	}
#endif

	delete[] in;
	in = output;
	dim = output_dim;
}
void NeuralNetwork::Add(float* in, unsigned int dim, float* biases) {
#if _USE_AVX == 1
	for (int i = 0; i < dim; i += AVX_FWORDSIZE) {
		__m256 avx_in = _mm256_loadu_ps(in + i);
		__m256 avx_bias = _mm256_loadu_ps(biases + i);
		__m256 res = _mm256_add_ps(avx_in, avx_bias);
		_mm256_storeu_ps(in + i, res);
	}
#else
	for (int i = 0; i < dim; i++) {
		in[i] += biases[i];
	}
#endif
}
void NeuralNetwork::RandomWeights() {
	for (int l = 0; l < weightlayer.amount_weightlayers; l++) {
		for (int o = 0; o < weightlayer.weightlayersizes[l].x; o++) {
			for (int i = 0; i < weightlayer.weightlayersizes[l].y; i++) {
				weights[l][o * weightlayer.weightlayersizes[l].y + i] = Random::RandomFloat(-1.0f, 1.0f);
			}
		}
	}
}
void NeuralNetwork::RandomBiases() {
	for (int l = 0; l < layer.amount_layers; l++) {
		for (int n = 0; n < layer.layersizes[l]; n++) {
			biases[l][n] = Random::RandomFloat(-1.0f, 1.0f);
		}
	}
}
void NeuralNetwork::PrintOutput() {
	if (output != nullptr) {
		std::cout << "Output: " << std::endl;
		for (int i = 0; i < layer.layersizes[layer.amount_layers - 1]; i++) {
			printf("%.2f ", output[i]);
		}
		printf("\n");
	}
}



int main() {
	srand(time(NULL));

	NeuralNetwork nn;
	NeuralNetwork::InputParameter ip;
	ip.amount_layers = 5;
	ip.layersizes = new unsigned int[5];
	ip.layersizes[0] = 200;
	ip.layersizes[1] = 1000;
	ip.layersizes[2] = 10000;
	ip.layersizes[3] = 1000;
	ip.layersizes[4] = 8;

	ip.activation_functions.push_back(ActivationFunction::Linear);
	ip.activation_functions.push_back(ActivationFunction::Linear);
	
	Clock c;

	c.Start();
	nn.Compile(ip);
	nn.RandomWeights();
	std::cout << "Compilation: " << c.ElapsedMilliseconds() << "ms" << std::endl;

	float* input = new float[200];
	for (int i = 0; i < 200; i++) {
		input[i] = 1.0f;
	}

	c.Start();
	nn.Run(input, 200);
	std::cout << "Running: " << c.ElapsedMilliseconds() << "ms" << std::endl;
	nn.PrintOutput();

	return 0;
}
