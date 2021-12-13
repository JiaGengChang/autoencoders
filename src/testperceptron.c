#include <perceptron.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_ITER 50 
#define NUM_INPUTS 10
#define NUM_TRAIN_RECORDS 768
#define NUM_TEST_RECORDS 256

int main()
{
	char* train_dir = "../data/parity_10_inputs.csv";
	char* test_dir = "../data/parity_10_inputs_t.csv";

	Perceptron per = { .numFeatures=NUM_INPUTS, .learningRate=1e-5 };
	initialize(&per, 0.1f);
	printParams(&per);
	
	/*allocate train array*/
	float signal[NUM_TRAIN_RECORDS];
	float **inputs;
	inputs = malloc(NUM_TRAIN_RECORDS * sizeof *inputs);
	int i=0;
	for (;i<NUM_TRAIN_RECORDS;++i)
		inputs[i] = malloc(NUM_INPUTS * sizeof * inputs[i]);

	parseinput(train_dir, inputs, signal, NUM_INPUTS);
	//echoinput(inputs, signal, NUM_INPUTS, NUM_TRAIN_RECORDS);

	/*training*/
	train(&per, NUM_ITER, inputs, signal, NUM_TRAIN_RECORDS);

	/*allocate test array*/
	float signal_test[NUM_TEST_RECORDS];
	float **inputs_test;
	inputs_test = malloc(NUM_TEST_RECORDS * sizeof * inputs_test);
	for (i=0;i<NUM_TEST_RECORDS;++i)
		inputs_test[i] = malloc(NUM_INPUTS * sizeof * inputs_test[i]);

	parseinput(test_dir, inputs_test, signal_test, NUM_INPUTS);

	/*testing*/
	test(&per, inputs_test, signal_test, NUM_TEST_RECORDS);


	/* deallocate */
	for (i=0; i<NUM_TRAIN_RECORDS;i++)
	{
		free(inputs[i]);
	}
	free(inputs);

	for (i=0; i<NUM_TEST_RECORDS;i++)
	{
		free(inputs_test[i]);
	}
	free(inputs_test);

	return 0;

}
