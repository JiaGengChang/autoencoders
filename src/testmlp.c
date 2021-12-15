#include <utils.h>

void train(int numIterations)
{
	int nIter;

}


int main()
{
	static double input[] = {
		1,0,0,0,0,0,0,0,
		0,1,0,0,0,0,0,0,
		0,0,1,0,0,0,0,0,
		0,0,0,1,0,0,0,0,
		0,0,0,0,1,0,0,0,
		0,0,0,0,0,1,0,0,
		0,0,0,0,0,0,1,0,
		0,0,0,0,0,0,0,1
	};
	
	Matrix *m = mat_new(8,8, input);

	print_mat(m); 

	randomize(m, 0, 1);	

	print_mat(m);

	print_mat(mat_dot(m,m));

	return 0;
	
}
