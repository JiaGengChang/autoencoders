#include <perceptron.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

int main()
{
	char *train_dir = "data/step_function/9.csv";
	char *test_dir = "data/step_function/9t.csv";

	Perceptron per = { .numFeatures=NUM_INPUTS, .learningRate=1e-5 };
	initialize(&per, 0.1f);
	printParams(&per);
	
	/*allocate train array*/
	float signal[NUM_TRAIN_RECORDS];
	float inputs[NUM_TRAIN_RECORDS][NUM_INPUTS];

	parseInput(train_dir, inputs, signal);
	echoInput(inputs, NUM_TRAIN_RECORDS, signal);

	/*training*/
	train(&per, NUM_ITER, inputs, signal);

	/*allocate test array*/
	float signal_test[NUM_TEST_RECORDS];
	float inputs_test[NUM_TEST_RECORDS][NUM_INPUTS];

	parseInput(test_dir, inputs_test, signal_test);

	/*testing*/
	test(&per, inputs_test, signal_test);

	return 0;

}
