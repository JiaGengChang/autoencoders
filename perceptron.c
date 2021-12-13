#include <./perceptron.h>

// activation function - unit threshold function
extern inline float activate(float value)
{
	return (value < 0.0f) ? 0.0f : 1.0f;
}

// get a random float between [-0.5a, 0.5a]
extern inline float runif(float a)
{
	return (float)rand()/(float)(RAND_MAX/a) - 0.5*a;
}

//perceptron functions ============================

float forwardPass(Perceptron *per, float inputs[])
{
	float value = 0.0f;
	float prediction;

	// v = wi * xi + b
	for (int i = 0; i < per->numFeatures; ++i){
		value += per->weights[i] * inputs[i];
	}
	value += per->bias;

	// get activation
	prediction = activate(value);

	return prediction;
}

void backwardPass(Perceptron *per, float prediction, float inputs[], float signal) {
	// update weights
	int i;
	for (i=0; i < per->numFeatures; ++i){
		per->weights[i] += per->learningRate * inputs[i] * (signal - prediction);
	}
	// update bias
	per->bias += per->learningRate * inputs[i] * (signal - prediction);

}

void initialize(Perceptron *per)
{
	float a = 0.1f; //random value between -0.5a and 0.5a
	int i;
	for (i=0; i<per->numFeatures; ++i)
	{
		per->weights[i] = runif(a); 
	}
	per->bias = runif(a);
}

void updateConfusionMatrix(float guess, float truth, int *pTP,  int *pFP, int *pTN, int *pFN)
{
   	if (abs(guess-truth) < 1e-3) {
		if (abs(truth-1.0f) < 1e-3) { (*pTP)++; printf("TP\n");}
		else { (*pFP)++; printf("FP\n");}
	}
	else 
	{
		if (abs(truth-1.0f) < 1e-3) { (*pTN)++; printf("TN\n");} 
		else { (*pFN)++; printf("FN\n");}
	}
}

void train(Perceptron *per, int numIterations, float **inputs, float *signal, int numExamples)
{
	int TP, FP, TN, FN;
	int nIter=0;

	// for each epoch
	for (; nIter < numIterations; ++nIter)
	{
		int nExample=0;
		TP = 0; FP = 0; TN = 0; FN = 0;
		
		// iterate over training records
		for (; nExample < numExamples; ++nExample)
		{
			float prediction;
			//predict
			prediction = forwardPass(per, inputs[nExample]);
			//keep track of scores
			updateConfusionMatrix(prediction, signal[nExample], &TP, &FP, &TN, &FN);
			//update parameters
			backwardPass(per, prediction, inputs[nExample], signal[nExample]);
		}

		// metrics
		printf("nIter: %d, TP: %d, FP: %d, TN: %d, FN: %d\n", nIter, TP, FP, TN, FN);
	}
}

void printParams(Perceptron *per)
{
	printf("bias: %f \n", per->bias);
	printf("weights: ");
	int i;
	for (i=0; i<per->numFeatures; ++i)
	{
		printf("%f ", per->weights[i]);
	}
	printf("\n");
}

// IO functions ==============================

void parseinput(char *fn, float **inputs, float signal[])
{
	FILE* fp = fopen(fn,"r");
	char buff[1024];
	int nRecord = 0;
	while(fgets(buff, 1024, fp))
	{
		char *field = strtok(buff, ",");
		int field_count = 0;
		while (field)
		{
			if (field_count==10)
				signal[nRecord] = atof(field);
			else 
				inputs[nRecord][field_count] = atof(field);

			field = strtok(NULL, ",");
			++field_count;
		}			
		++nRecord;
	}
	fclose(fp);
}

void echoinput(float **inputs, float *signal, const int NUM_TRAIN_RECORDS)
{
	int i;
	for (i=0;i<NUM_TRAIN_RECORDS;++i)
	{
		int j=0;
		printf("inputs: ");
		for (;j<10;++j)
		{
			printf("%f ",inputs[i][j]);
		}
		printf("\nsignal: %f\n", signal[i]);
	}
}
