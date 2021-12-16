#include <utils.h>


void mat_apply(Matrix *A, double (*function)(double), Matrix *B)
{
	size_t i,j;
	for (i=0; i<A->nrows; ++i)
	{
		double x, z;
		for (j=0; j<A->ncols; ++j)
		{
			x = mat_get(A, i, j);
			z = function(x);
			mat_set(B, i, j, z);
		}
	}
}

Matrix* mat_new(const size_t n1, const size_t n2, double input[])
{
	Matrix *m = malloc(sizeof(Matrix) + n1*n2*sizeof(double));
	m->nrows = n1;
	m->ncols = n2;
	size_t i,j;
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


double mat_get(const Matrix *m, const size_t i, const size_t j)
{
	return m->data[i * m->ncols + j];
}

void mat_set(Matrix *m, const size_t i, const size_t j, double v)
{
	m->data[i * m->ncols + j] = v;
	if (m->data[i * m->ncols + j] != v) {
		printf("Value not changed\n");
	}
}

void mat_dot(Matrix *A, Matrix *B, Matrix *C)
{
	size_t i,j,k;
	for (i=0; i < A->nrows; ++i){
		for (j=0; j < B->ncols; ++j){
			mat_set(C,i,j,0.0f);
			for (k=0; k < A->ncols; ++k){
				mat_set(C, i, j, mat_get(C, i,j) + mat_get(A,i,k) * mat_get(B,k,j));
			}
		}
	}
}

void mat_add(Matrix *A, Matrix *B, Matrix *C)
{	
	size_t i,j;
	for (i=0; i < A->nrows; ++i){
		for (j=0; j < B->ncols; ++j){
			mat_set(C, i, j, mat_get(A,i,j) + mat_get(B,i,j));
		}
	}
}

//hadamard product
void mat_mult(Matrix *A, Matrix *B, Matrix *C)
{	
	size_t i,j;
	for (i=0; i < A->nrows; ++i){
		for (j=0; j < B->ncols; ++j){
			mat_set(C, i, j, mat_get(A,i,j) * mat_get(B,i,j));
		}
	}
}

Matrix* mat_transpose(Matrix *m)
{
	size_t i,j;
	Matrix *mt = mat_new(m->ncols, m->nrows, NULL);
	for (i=0; i < m->nrows; ++i){
		for (j=0; j < m->ncols; ++j){
			mat_set(mt, j, i, mat_get(m,i,j));
		}
	}
	return mt;
}

void mat_free(Matrix *m)
{
	free(m);
}

void scalar_add(Matrix *A, const double s, Matrix *B)
{	
	size_t i,j;
	for (i=0; i < A->nrows; ++i){
		for (j=0; j < A->ncols; ++j){
			mat_set(B, i, j, mat_get(A,i,j) + s);
		}
	}
}

void scalar_mult(const Matrix *A, const double s, Matrix *B)
{	
	size_t i,j;
	for (i=0; i < A->nrows; ++i){
		for (j=0; j < A->ncols; ++j){
			mat_set(B, i, j, mat_get(A,i,j) * s);
		}
	}
}


void print_mat(const Matrix *m)
{
	size_t i,j;
	printf("Dimensions: (%lu, %lu)\n", m->nrows, m->ncols);
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
