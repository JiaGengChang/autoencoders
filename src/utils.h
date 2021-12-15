#include <math.c>
#include <stdio.c>

#ifndef utils_h
#define utils_h

typedef struct {
	size_t nrows;
	size_t ncols;
	double array[nrows*ncols];
} Matrix;



#endif
