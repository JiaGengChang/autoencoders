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
	float a = 2.0f; //random value between -0.5a and 0.5a
	int i;
	for (i=0; i<per->numFeatures; ++i)
	{
		per->weights[i] = runif(a); 
	}
	per->bias = runif(a);
}

void updateConfusionMatrix(int guess, int truth, int *pTP,  int *pFP, int *pTN, int *pFN)
{
   	if ((int) guess == (int) truth) {
		if ((int) truth == 1) { (*pTP)++;}
		else { (*pTN)++;} }
	else 
	{
		if ((int) truth == 1) { (*pFN)++;} 
		else { (*pFP)++;}
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
			float activation;
			int prediction;
			//predict
			activation = forwardPass(per, inputs[nExample]);
			prediction = (int) (activation + 0.5);
			//keep track of scores
			updateConfusionMatrix(prediction, (int) signal[nExample], &TP, &FP, &TN, &FN);
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

void parseinput(char *fn, float **inputs, float signal[], const int NUM_INPUTS)
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
			if (field_count==NUM_INPUTS)
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

void echoinput(float **inputs, float *signal, const int NUM_INPUTS, const int NUM_TRAIN_RECORDS)
{
	int i;
	for (i=0;i<NUM_TRAIN_RECORDS;++i)
	{
		int j=0;
		printf("inputs: ");
		for (;j<NUM_INPUTS;++j)
		{
			printf("%d ",(int) inputs[i][j]);
		}
		printf("\nsignal: %d\n", (int) signal[i]);
	}
}
