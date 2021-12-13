#include <perceptron.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_ITER 100 
#define NUM_TRAIN_RECORDS 3 

int main()
{
	Perceptron per = { .numFeatures=10, .learningRate=1e-3};
	initialize(&per);
	printParams(&per);
	
	/*allocate train array*/
	float signal[NUM_TRAIN_RECORDS];
	float **inputs;
	inputs = malloc(NUM_TRAIN_RECORDS * sizeof *inputs);
	int i=0;
	for (;i<10;++i)
		inputs[i] = malloc(11 * sizeof * inputs[i]);

	parseinput("input.csv", inputs, signal);

	/*training*/
	echoinput(inputs, signal, NUM_TRAIN_RECORDS);
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
