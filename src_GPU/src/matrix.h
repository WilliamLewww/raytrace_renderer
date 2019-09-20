#pragma once
#include "tuple.h"

/*

[   0   1   2   3  ]
[   4   5   6   7  ]
[   8   9  10  11  ]
[  12  13  14  15  ]

*/

struct Matrix {
	float* data;

	int rowCount;
	int columnCount;
};

void setDataMatrix(Matrix* matrix, int row, int column, float value);
float* getDataMatrixPointer(Matrix* matrix, int row, int column);
Matrix createMatrix(int rowCount, int columnCount);
Matrix createIdentityMatrix(int rowColumnCount);
Matrix operator*(Matrix lhs, Matrix rhs);
float determinate(Matrix matrix);
Matrix submatrix(Matrix matrix, int row, int column);
float matrixMinor(Matrix matrix, int row, int column);
float cofactor(Matrix matrix, int row, int column);
Matrix inverse(Matrix matrix);

__host__ __device__
Tuple operator*(Matrix lhs, Tuple rhs);

__host__ __device__
Matrix createMatrix(int rowCount, int columnCount, float* data);

__host__ __device__
float getDataMatrix(Matrix* matrix, int row, int column);