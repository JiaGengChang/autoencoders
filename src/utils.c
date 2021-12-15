#include <utils.h>


Matrix* mat_new(const size_t n1, const size_t n2, double input[])
{
	Matrix *m = malloc(sizeof(Matrix) + n1*n2*sizeof(double));
	m->nrows = n1;
	m->ncols = n2;
	int i,j;
	if (input==NULL) {
		for (i=0; i<n1; ++i)
			for (j=0; j<n2; ++j)
				mat_set(m, i, j, 0.0f);
	}
	else {
		for (i=0; i<n1; ++i)
			for (j=0; j<n2; ++j)
				mat_set(m, i, j, input[i * n2 + j]);
	}
	return m;
}

void randomize(Matrix *m, double min, double max)
{
	size_t i,j;
	for (i=0; i<m->nrows;++i)
	{
		for (j=0; j<m->ncols;++j)
		{
			mat_set(m, i, j, min + (max-min)*rand()/RAND_MAX);
		}
	}
}

Matrix* mat_cpy(const Matrix *A)
{  
	size_t i,j,k;
	Matrix *B;
	double array_of_a[A->nrows * A->ncols];

	for (i=0; i < A->nrows; ++i){
		for (j=0; j < A->ncols; ++j){
			array_of_a[i * A->ncols + j] = mat_get(A,i,j);
		}
	}

	B = mat_new(A->nrows, A->ncols, array_of_a);
	
	return B;
}



double mat_get(const Matrix *m, const size_t i, const size_t j)
{
	return m->data[i * m->ncols + j];
}

void mat_set(Matrix *m, const size_t i, const size_t j, double v)
{
	m->data[i * m->ncols + j] = v;
}

Matrix* mat_dot(const Matrix *A, const Matrix *B)
{
	size_t i,j,k;
	Matrix *C;
	C = mat_new(A->nrows, B->ncols,NULL);

	for (i=0; i < A->nrows; ++i){
		for (j=0; j < B->ncols; ++j){
			mat_set(C,i,j,0.0f);
			for (k=0; k < A->ncols; ++k){
				mat_set(C, i, j, mat_get(C, i,j) + mat_get(A,i,k) * mat_get(B,k,j));
			}
		}
	}
	return C;
}

Matrix* mat_add(const Matrix *A, const Matrix *B)
{	
	size_t i,j;
	Matrix *C;
	C = mat_new(A->nrows, A->ncols, NULL);

	for (i=0; i < A->nrows; ++i){
		for (j=0; j < B->ncols; ++j){
			mat_set(C, i, j, mat_get(A,i,j) + mat_get(B,i,j));
		}
	}
	return C;
}

Matrix* scalar_add(const Matrix *A, const double s)
{	
	size_t i,j;
	Matrix *B;
	B = mat_new(A->nrows, A->ncols, NULL);

	for (i=0; i < A->nrows; ++i){
		for (j=0; j < A->ncols; ++j){
			mat_set(B, i, j, mat_get(A,i,j) + s);
		}
	}
	return B;
}

Matrix* scalar_mult(const Matrix *A, const double s)
{	
	size_t i,j;
	Matrix *B;
	B = mat_new(A->nrows, A->ncols, NULL);

	for (i=0; i < A->nrows; ++i){
		for (j=0; j < A->ncols; ++j){
			mat_set(B, i, j, mat_get(A,i,j) * s);
		}
	}
	return B;
}


void print_mat(const Matrix *m)
{
	int i,j;
	for (i=0; i<m->nrows; ++i)
	{ 
		for (j=0; j<m->ncols; ++j)
		{
			printf("%.2f ", mat_get(m,i,j));
		}
		printf("\n");
	}
	printf("\n");
}
