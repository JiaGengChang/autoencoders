#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifndef mlp_h
#define mlp_h

#define NUM_INPUT 8 //number of input units
#define NUM_HIDDEN 3 //number of hidden units
#define NUM_OUTPUT 8 //number of output units
#define NUM_PARAMS NUM_INPUT*NUM_HIDDEN+NUM_INPUT*NUM_OUTPUT //number of trainable parameters
#define NUM_TRAIN 8 //size of training data, no test data here

typedef struct {
	double input;
	double activation;
	double derivative;
	double delta;
} Unit;

typedef struct {
	Unit input[NUM_INPUT]; 
	Unit hidden[NUM_HIDDEN];
	Unit output[NUM_OUTPUT];
	double weights[NUM_PARAMS]; //best to be a 1d array as connections may not be symmetric
	double biases[NUM_HIDDEN + NUM_OUTPUT]
} MultiLayerPerceptron;


void resetUnit(Unit *unit);
void resetNetwork(MultiLayerPerceptron *mlp);
void activate(Unit *unit);
void initialize(MultiLayerPerceptron *mlp, double plus_minus_half_of);
void forwardPass(MultiLayerPerceptron *mlp, const double data[NUM_INPUT]);
void backwardPass(MultiLayerPerceptron *mlp, const double signal[NUM_OUTPUT], const double learningRate);
void train(MultiLayerPerceptron *mlp, const double input[NUM_TRAIN][NUM_INPUT], int numIterations, const double learningRate);
void test(MultiLayerPerceptron *mlp, const double input[NUM_TRAIN][NUM_INPUT]);
void error(MultiLayerPerceptron *mlp, const double input[NUM_TRAIN][NUM_INPUT], double *error);
void evaluate(MultiLayerPerceptron *mlp, const double input[NUM_TRAIN][NUM_INPUT], unsigned int *ncorrect);
void parseInput(char *fn, double input[NUM_TRAIN][NUM_INPUT]);
void echoInput(double input[NUM_TRAIN][NUM_INPUT]);
void printParams(MultiLayerPerceptron *mlp);

#endif
