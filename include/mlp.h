#include <math.h>

#ifndef mlp_h
#define mlp_h

#define NUM_INPUT 8 //number of input units
#define NUM_HIDDEN 3 //number of hidden units
#define NUM_OUTPUT 8 //number of output units
#define NUM_PARAMS NUM_INPUT*NUM_HIDDEN+NUM_INPUT*NUM_OUTPUT //number of trainable parameters
#define learningRate 1e-4

#define NUM_TRAIN 8 //size of training data, no test data here

typedef struct {
	float input;
	float activation;
	float delta;
	float bias;
} Unit;

typedef struct {
	struct Unit input[NUM_INPUT]; 
	struct Unit hidden[NUM_HIDDEN];
	struct Unit output[NUM_OUTPUT];
	float weights[NUM_PARAMS]; //best to be a 1d array as connections may not be symmetric
} MultiLayerPerceptron;


void resetUnit(Unit *unit);
void resetNetwork(MultiLayerPerceptron *mlp);
void activate(Unit *unit);
void initializeRandomWeights(MultiLayerPerceptron *mlp, float plus_minus_half_of);
void forwardPass(MultiLayerPerceptron *mlp, float data[NUM_INPUT]);
void backwardPass(MultiLayerPerceptron *mlp, float signal[NUM_OUTPUT])
void train(MultiLayerPerceptron *mlp, float input[NUM_TRAIN][NUM_INPUT])

#endif
