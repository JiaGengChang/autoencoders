#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef perceptron_h
#define perceptron_h

#define MAX_N_FEATURES 100

typedef struct {
	int numFeatures;
	float weights[MAX_N_FEATURES];
	float bias;
	float learningRate;	
} Perceptron;

float forwardPass(Perceptron *per, float inputs[]);
void backwardPass(Perceptron *per, float prediction, float inputs[], float signal);
void initialize(Perceptron *per);
void updateConfusionMatrix(float guess, float truth, int *pTP,  int *pFP, int *pTN, int *pFN);
void train(Perceptron *per, int numIterations, float **inputs, float *signal, int numExamples);
void printParams(Perceptron *per);
void parseinput(char *fn, float **inputs, float signal[]);
void echoinput(float **inputs, float *signal, const int NUM_TRAIN_RECORDS);
#endif
