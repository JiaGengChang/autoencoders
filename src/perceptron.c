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

void initialize(Perceptron *per, float plus_minus_half_of)
{
	//initialize weights to random value between [-0.5,0.5]*plus_minus_half_of
	int i;
	for (i=0; i<per->numFeatures; ++i)
	{
		per->weights[i] = runif(plus_minus_half_of); 
	}
	per->bias = runif(plus_minus_half_of);
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

void updateNetworkError(float prediction, float signal, float *pError)
{
	*pError += 0.5*pow(prediction - signal,2.0f);
}

void train(Perceptron *per, int numIterations, float **inputs, float *signal, int numExamples)
{
	int TP, FP, TN, FN; //confusion matrix
	float error; //network error
	int nEpoch=0;

	// for each epoch
	for (; nEpoch < numIterations; ++nEpoch)
	{
		int nExample=0;
		TP = 0; FP = 0; TN = 0; FN = 0;
		error = 0;
		
		// iterate over training records
		for (; nExample < numExamples; ++nExample)
		{
			float activation;
			int prediction;
			//network output
			activation = forwardPass(per, inputs[nExample]);
			//round network output to nearest integer
			prediction = (int) (activation + 0.5);
			//log performance
			updateConfusionMatrix(prediction, (int) signal[nExample], &TP, &FP, &TN, &FN);
			updateNetworkError(prediction, signal[nExample], &error);
			//update parameters
			backwardPass(per, prediction, inputs[nExample], signal[nExample]);
		}

		// metrics
		printf("Epoch %d, TP: %d, FP: %d, TN: %d, FN: %d, error: %.2f\n", nEpoch+1, TP, FP, TN, FN, error);
	}
}

void test(Perceptron *per, float **inputs, float *signal, int numExamples)
{
	int TP=0, FP=0, TN=0, FN=0, nExample=0;
	float error=0.0f;
	for (; nExample < numExamples; ++nExample)
	{
		float activation;
		int prediction;
		activation = forwardPass(per, inputs[nExample]);
		prediction = (int) (activation + 0.5);
		updateConfusionMatrix(prediction, (int) signal[nExample], &TP, &FP, &TN, &FN);
		updateNetworkError(prediction, signal[nExample], &error);
		//no update of weights
	}
	// metrics
	printf("Epoch TT, TP: %d, FP: %d, TN: %d, FN: %d, error: %.2f\n", TP, FP, TN, FN, error);
	
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

// debug functions ===========================
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
