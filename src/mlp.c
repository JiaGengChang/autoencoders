#include <./mlp.h>

extern inline float sigmoid(float input)
{
	return 1 / (1 + exp(-input));
}
extern inline float gradient(float input)
{
	return input * (1 - input);
}
extern inline runif(float a)
{
	return (float)rand()/(float)(RAND_MAX/a) - 0.5*a;
}

void resetUnit(Unit *unit)
{
	unit->input = 0.0f;
	unit->activation = 0.0f;
	unit->derivative = 0.0f;
	unit->bias = 1.0f;
}

void initializeRandomWeights(MultiLayerPerceptron *mlp, float plus_minus_half_of)
{
	int w;
	for (w=0;w<NUM_PARAMS;++w)
	{
		mlp->weights[w] = runif(plus_minus_half_of);
	}
}

void resetUnits(MultiLayerPerceptron *mlp)
{
	int i,j,k;
	for (i=0;i<NUM_INPUT;++i)
		resetUnit(&(mlp->input[i]));
	for (j=0;j<NUM_HIDDEN;++j)
		resetUnit(&(mlp->hidden[j]));
	for (k=0;k<NUM_OUTPUT;++k)
		resetUnit(&(mlp->output[k]));
}

void activate(Unit *unit)
{
	unit->activation = sigmoid(unit->input) + unit->bias;
	unit->derivative = sigmoid(unit->input) * (1 - sigmoid(unit->input));
}

void forwardPass(MultiLayerPerceptron *mlp, float data[NUM_INPUT])
{
	resetUnits(mlp);
	int i,j,k;
	
	for (i=0;i<NUM_INPUT;++i)
	{
		mlp->input[i].input = data[i];
		activate(&(mlp->input[i]));
	}

	/*input-hidden weights*/
	for (j=0;j<NUM_HIDDEN;++j)
	{
		for (i=0;i<NUM_INPUT;++i)
			mlp->hidden[j].input += mlp->input[i].input * mlp->weights[i*NUM_HIDDEN + j];
		activate(&(mlp->hidden[j]));
	}

	/*hidden-output weights*/
	for (k=0;k<NUM_OUTPUT;++k)
	{
		for (j=0;j<NUM_HIDDEN;++j)
			mlp->output[k].input += mlp->hidden[j] * mlp->weights[NUM_INPUT*NUM_HIDDEN + j*NUM_OUTPUT + k];
		activate(&(mlp->output[k]));
	}

}

void backwardPass(MultiLayerPerceptron *mlp, float signal[NUM_OUTPUT])
{
	int i,j,k;

	for (k=0;k<NUM_OUTPUT;++k)
	{
		/*hidden-output deltas*/
		mlp->output[k].delta  = mlp->output[k].derivative * (mlp->output[k].activation - signal[k]);
		for (j=0;j<NUM_HIDDEN;++j)
		{
			/*update hidden-output weights*/
			mlp->weights[NUM_INPUT*NUM_HIDDEN + j*NUM_OUTPUT + k] -= learningRate * mlp->output[k].delta * mlp->output[k].activation;
			/*input-hidden deltas*/
			mlp->hidden[j].delta += mlp->output[j].derivative * mlp->output[k].delta;
		}
	}

	for (j=0;j<NUM_HIDDEN;++j)
	{
		for (i=0;i<NUM_INPUT;++i)
		{
			/*update input-hidden weights*/
			mlp->weights[i*NUM_HIDDEN + j] -= learningRate * mlp->hidden[j].delta * mlp->hidden[j].activation;
		}
	}
	
}

float scoreNetwork(MultiLayerPerceptron *mlp, const float input[NUM_TRAIN][NUM_INPUT])
{
	int nTrain,nInput;
	float error=0.0f;
	//binary cross-entropy
	for (nTrain=0;nTrain<NUM_TRAIN;++nTrain)
	{
		for(nInput=0;nInput<NUM_INPUT;++NUM_INPUT)
		{
			error += input[nTrain][nInput] * log(mlp->output[nInput].activation);
		}
	}
	return error;
}

void train(MultiLayerPerceptron *mlp, const float input[NUM_TRAIN][NUM_INPUT], int numIterations)
{
	int nTrain, nIter;

	for (nIter=0; nIter<numIterations; ++nIter)
	{
		for (nTrain=0; nTrain<NUM_TRAIN; ++nTrain)
		{
			//evalute
			forwardPass(mlp, input[nTrain]);
			// update parameters
			backwardPass(mlp, input[nTrain]);
		}
		printf("Iteration %d, %.2f", nIter, scoreNetwork(mlp, input));
	}

	printf("Finished, %.2f", scoreNetwork(mlp, input));

}


void parseInput(char *fn, float input[NUM_TRAIN][NUM_INPUT])
{
	FILE *fp = fopen(fn, "r");
	char buff[1024];
	int nRecord = 0;
	while (fgets(buff, 1024, fp))
	{
		char *field = strtok(buff,",");
		int field_count = 0;
		while (field)
		{
			inputs[nRecord][field_count] = atof(field);
			field = strtok(NULL, ",");
			++field_count;
		}
		++nRecord;
	}
	fclose(fp);
}


void echoInput(float input[NUM_TRAIN][NUM_INPUT])
{
	int nTrain,nInput;
	for (nTrain=0; nTrain<NUM_TRAIN; ++nTrain)
	{
		for (nInput=0; nInput<NUM_INPUT; ++nInput)
		{
			printf(" %d", input[nTrain][nInput];
		}
		printf("\n");
	}
}
