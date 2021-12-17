#include <utils.h>

double sigmoid(double x)
{
	return (1/(1 + exp(-x)));
}

double dsigmoid(double x) //derivative of sigmoid
{
	return (sigmoid(x)*(1-sigmoid(x)));
}

double logits_to_probability(double x)
{
	double odds = exp(x);
	double probability = odds / (1+odds);
}

int main()
{
	//the 8 cases to learn
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
	static double i_init[] = {0,0,0,0,0,0,0,0,1};
	static double j_init[] = {0,0,0,1};
	static double k_init[] = {0,0,0,0,0,0,0,0};
	static const double learningRate = 1.0;
	static const double momentum = 0.0;
	static const size_t numIterations = 1e4;
	static const size_t printFirst = 1e4;
	static const size_t printEvery = 1e2;

	printf("#params: learningRate %.3f, momentum %.3f, numIterations %lu, printEvery %lu, printFirst %lu\n", learningRate, momentum, numIterations, printEvery, printFirst);

	int debug = 0; //whether to print matrices. 1 prints weights only. 2 prints vectors
	
	Matrix *t = mat_new(8,8, input); // targets
	if (debug) {printf("Input matrix: \n"); print_mat(t);}

	Matrix *zi = mat_new(1,9,i_init);//input layer. 8 units + 1 bias
	Matrix *xj = mat_new(1,4,j_init);//hidden layer. 3 units + 1 bias 
	Matrix *zj = mat_new(1,4,j_init);
	Matrix *dj = mat_new(1,4,j_init);
	Matrix *xk = mat_new(1,8,k_init);//output layer. no bias units
	Matrix *zk = mat_new(1,8,k_init);
	Matrix *dk = mat_new(1,8,k_init);

	Matrix *W_ij = mat_new(9,4,NULL); // bias "unit" of hidden layer has no incoming weights, so its actually 9x3 not 9x4. I will set col 4 of W_ij to 0 later
	Matrix *W_jk = mat_new(4,8,NULL);
	Matrix *dW_ij = mat_new(9,4,NULL); //amount to adjust weights by. same here regarding 9x4 vs 9x3
	Matrix *dW_jk = mat_new(4,8,NULL);

	randomize(W_ij, -0.1f, 0.1f);
	randomize(W_jk, -0.1f, 0.1f);

	//track previous changes to weights 
	//start off as 0 matrices
	Matrix *dW_ij_prev = mat_new(9,4,NULL); 
	Matrix *dW_jk_prev = mat_new(4,8,NULL);

	//set 4th column of W_ij to 0
	//to turn off weights from input layer to bias unit of hidden layer
	size_t i;
	for (i=0; i<9; ++i) {mat_set(W_ij, i, 3, 0.0f);}
	
	size_t nIter, nInput, nUnit;
	for (nIter=0; nIter<numIterations; ++nIter)
	{
		//re-order training data
		shuffle(t, 3);

		double batch_loss = 0.0f; //binary cross-entropy loss

		//reconstruct current input
		for (nInput=0; nInput<8; ++nInput)
		{
			//initialize input layer of mlp
			for (nUnit=0; nUnit<8; ++nUnit)	{mat_set(zi, 0, nUnit, mat_get(t, nInput, nUnit));}

			//encode
			mat_dot(zi, W_ij, xj); //1x9 9x4 1x4
			mat_apply(xj, sigmoid, zj); //1x4
			mat_apply(xj, dsigmoid, dj); //1x4

			//modify bias activation and derivative
			mat_set(zj, 0, 3, 1.0f);
			mat_set(dj, 0, 3, 0.0f);

			//decode
			mat_dot(zj, W_jk, xk); //1x4 4x8 1x8
			mat_apply(xk, sigmoid, zk); //1x8
			mat_apply(xk, dsigmoid, dk); //1x8

			if (debug==2 && nIter%printEvery==0)
			{
				printf("Input layer activation\n"); 
				print_mat(zi);
				printf("Hidden layer activation\n");
				print_mat(zj);
				printf("Output layer activation\n");
				print_mat(zk);
			}

			//cross-entropy loss for one training example
			//- 1/N * sum over k of {t_k * log (p_k)}
			Matrix *loss_vec = mat_new(1, 8, zk->data);
			mat_apply(loss_vec, logits_to_probability, loss_vec);
			mat_apply(loss_vec, log, loss_vec);
			mat_mult(loss_vec, zi, loss_vec);
			scalar_mult(loss_vec, -1, loss_vec);
			double loss_scalar = 0.0f;
			for (i=0; i<8; ++i) {loss_scalar += mat_get(loss_vec, 0, i);}
			mat_free(loss_vec);

			//update batch loss
			batch_loss += loss_scalar/8;

			//back-propagation
			double delta_k_array[8];
			double t_k, z_k, d_k;
			for (nUnit=0; nUnit<8; ++nUnit)
			{
				t_k = mat_get(t,nInput,nUnit);
				z_k = mat_get(zk,0,nUnit); 
				d_k = mat_get(dk,0,nUnit);
				delta_k_array[nUnit] = -(t_k/z_k - (1-t_k)/(1-z_k)) * d_k; //k-dependent part of dE/dW_jk
			}
			/*Update W_jk*/
			Matrix *delta_k = mat_new(1,8,delta_k_array);//1x8
			Matrix *zj_T = mat_transpose(zj); //4x1
			mat_dot(zj_T, delta_k, dW_jk); //4x1, 1x8 -> 4x8

			scalar_mult(dW_jk_prev, momentum, dW_jk_prev);//4x8
			mat_add(dW_jk, dW_jk_prev, dW_jk);//4x8
			mat_assign(dW_jk_prev, dW_jk);//dW_jk_prev := dW_jk

			scalar_mult(dW_jk, -learningRate, dW_jk); //4x8
			mat_add(W_jk, dW_jk, W_jk); //update W_jk
			
			/*Update W_ij*/
			Matrix *delta_j = mat_new(1,4,NULL); //1x4
			Matrix *W_jk_T = mat_transpose(W_jk); //8x4
			mat_dot(delta_k, W_jk_T, delta_j); //1x8, 8x4 -> 1x4
			mat_mult(delta_j, dj, delta_j); //1x4, 1x4 -> 1x4 hadamard product
			
			Matrix *zi_T = mat_transpose(zi); //9x1
			mat_dot(zi_T, delta_j, dW_ij); // 9x1, 1x4 -> 9x4
			
			scalar_mult(dW_ij_prev, momentum, dW_ij_prev); //9x4
			mat_add(dW_ij, dW_ij_prev, dW_ij); //9x4
			mat_assign(dW_ij_prev, dW_ij);// dW_ij_prev := dW_ij
			
			scalar_mult(dW_ij, -learningRate, dW_ij); // 9x4
			mat_add(W_ij, dW_ij, W_ij); // update W_ij

			//remove weights going into bias
			for (i=0; i<9; ++i) {mat_set(W_ij, i, 3, 0.0f);} 

			//free temporary matrices
			mat_free(delta_k);
			mat_free(zj_T);
			mat_free(delta_j);
			mat_free(zi_T);
		}
		//logging
		if (nIter < printFirst || nIter % printEvery == 0)
			printf("iteration %lu, loss %.8f\n", nIter, batch_loss);
	}
	//visualize weights
	if (debug==1)
	{
		printf("W_ij\n");
		print_mat(W_ij);
		printf("W_jk\n");
		print_mat(W_jk);
	}

	return 0;
	
}
