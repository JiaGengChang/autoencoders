#include <./mlp.h>

int main()
{
	float input[NUM_TRAIN][NUM_INPUT];
	char *datafile = "../data/one_hot.csv";
	parseInput(datafile, input);
	echoInput(input);

	return 0;

}

