#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef perceptron_h
#define perceptron_h

#define MAX_N_FEATURES 100
#define COUNT_OF(x) (sizeof(x)/sizeof(x[0]))
#define NUM_ITER 50 
#define NUM_INPUTS 9
#define NUM_TRAIN_RECORDS 384
#define NUM_TEST_RECORDS 128


typedef struct {
	int numFeatures;
	float weights[MAX_N_FEATURES];
	float bias;
	float learningRate;	
} Perceptron;

float forwardPass(Perceptron *per, float *inputs);
void backwardPass(Perceptron *per, float prediction, float *inputs, float signal);
void initialize(Perceptron *per, float plus_minus_half_of);
void updateConfusionMatrix(int guess, int truth, int *pTP,  int *pFP, int *pTN, int *pFN);
void updateNetworkError(float prediction, float signal, float *pError);
void train(Perceptron *per, int numIterations, float inputs[][NUM_INPUTS], float *signal);
void test(Perceptron *per, float inputs[][NUM_INPUTS], float *signal);
void printParams(Perceptron *per);
void parseInput(char *fn, float inputs[][NUM_INPUTS], float *signal);
void echoInput(float inputs[][NUM_INPUTS], float *signal);
#endif
