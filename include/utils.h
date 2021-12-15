#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef utils_h
#define utils_h

typedef struct {
	size_t nrows;
	size_t ncols;
	double data[];
} Matrix;

Matrix* mat_new(const size_t n1, const size_t n2, double input[]);
void randomize(Matrix *m, double min, double max);
Matrix* mat_cpy(const Matrix *A);
double mat_get(const Matrix *m, const size_t i, const size_t j);
void mat_set(Matrix *m, const size_t i, const size_t j, double v);
Matrix* mat_dot(const Matrix *matA, const Matrix *matB);
Matrix* mat_add(const Matrix *A, const Matrix *B);
Matrix* scalar_add(const Matrix *A, const double s);
Matrix* scalar_mult(const Matrix *A, const double s);
void print_mat(const Matrix *m);

#endif
