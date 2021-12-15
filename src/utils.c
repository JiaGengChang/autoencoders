#include <utils.h>


float mat_get(const Matrix *m, const size_t i, const size_t j)
{
	return m->data[i * m->rowlen + j]
}

void mat_set(Matrix *m, const size_t i, const size_t j, float v)
{
	m->data[i * m->rowlen + j] = v;
}

Matrix mat_dot(const Matrix *A, const Matrix *B)
{
	const size_t i,j,k;
	Matrix matC {.nrows=matA.nrows, .ncols=matB.ncols};

	for(i=0;i<M;i++){
		for(j=0;j<K;j++){
			mat_set(matC,i,j)=0.0f;
			for(k=0;k<N;k++){
				mat_acc(matC,i,j,mat_get(matA,i,k)*mat_get(matB,k,j);
			}
		}
	}

	return matC;
}
