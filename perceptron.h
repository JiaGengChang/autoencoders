#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef perceptron_h
#define perceptron_h

#define MAX_N_FEATURES 100
#define COUNT_OF(x) (sizeof(x)/sizeof(x[0]))

typedef struct {
	int numFeatures;
	float weights[MAX_N_FEATURES];
	float bias;
	float learningRate;	
} Perceptron;

float forwardPass(Perceptron *per, float inputs[]);
void backwardPass(Perceptron *per, float prediction, float inputs[], float signal);
void initialize(Perceptron *per, float plus_minus_half_of);
void updateConfusionMatrix(int guess, int truth, int *pTP,  int *pFP, int *pTN, int *pFN);
void updateNetworkError(float prediction, float signal, float *pError);
void train(Perceptron *per, int numIterations, float **inputs, float *signal, int numExamples);
void test(Perceptron *per, float **inputs, float *signal, int numExamples);
void printParams(Perceptron *per);
void parseinput(char *fn, float **inputs, float signal[], const int NUM_INPUTS);
void echoinput(float **inputs, float *signal, const int NUM_INPUTS, const int NUM_TRAIN_RECORDS);
#endif
