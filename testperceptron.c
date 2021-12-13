#include <perceptron.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_ITER 20 
#define NUM_TRAIN_RECORDS 2048 
#define NUM_INPUTS 11 

int main()
{
	Perceptron per = { .numFeatures=NUM_INPUTS, .learningRate=5e-4};
	initialize(&per);
	printParams(&per);
	
	/*allocate train array*/
	float signal[NUM_TRAIN_RECORDS];
	float **inputs;
	inputs = malloc(NUM_TRAIN_RECORDS * sizeof *inputs);
	int i=0;
	for (;i<NUM_TRAIN_RECORDS;++i)
		inputs[i] = malloc(NUM_INPUTS * sizeof * inputs[i]);

	parseinput("data_11_inputs.csv", inputs, signal, NUM_INPUTS);

	/*training*/
	echoinput(inputs, signal, NUM_INPUTS, NUM_TRAIN_RECORDS);
	train(&per, NUM_ITER, inputs, signal, NUM_TRAIN_RECORDS);

	/*deallocate train array*/
	/*deallocate tail*/
	for (i=1; i<NUM_TRAIN_RECORDS;i++)
	{
		free(inputs[i]);
	}
	/*deallocate head*/
	free(inputs);
	return 0;

}
