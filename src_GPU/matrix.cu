#include "matrix.h"

void setDataMatrix(Matrix* matrix, int row, int column, float value) {
	matrix->data[(row * matrix->columnCount) + column] = value;
}

float* getDataMatrixPointer(Matrix* matrix, int row, int column) {
	return &matrix->data[(row * matrix->columnCount) + column];
}

__host__ __device__
float getDataMatrix(Matrix* matrix, int row, int column) {
	return matrix->data[(row * matrix->columnCount) + column];
}

Matrix createMatrix(int rowCount, int columnCount) {
	Matrix matrix;

	matrix.rowCount = rowCount;
	matrix.columnCount = columnCount;

	matrix.data = new float[matrix.rowCount * matrix.columnCount];

	for (int y = 0; y < matrix.rowCount; y++) {
		for (int x = 0; x < matrix.columnCount; x++) {
			matrix.data[(y * matrix.columnCount) + x] = 0.0;
		}
	}

	return matrix;
}

__host__ __device__
Matrix createMatrix(int rowCount, int columnCount, float* data) {
	Matrix matrix;

	matrix.rowCount = rowCount;
	matrix.columnCount = columnCount;

	matrix.data = new float[matrix.rowCount * matrix.columnCount];
	memcpy(matrix.data, data, rowCount*columnCount*sizeof(float));

	return matrix;
}

Matrix createIdentityMatrix(int rowColumnCount) {
	Matrix matrix = createMatrix(rowColumnCount, rowColumnCount);
	for (int x = 0; x < rowColumnCount; x++) {
		setDataMatrix(&matrix, x, x, 1.0);
	}

	return matrix;
}

Matrix operator*(Matrix lhs, Matrix rhs) {
	Matrix matrix = createMatrix(lhs.rowCount, rhs.columnCount);
	for (int x = 0; x < lhs.rowCount; x++) {
		for (int y = 0; y < rhs.columnCount; y++) {
			for (int z = 0; z < lhs.columnCount; z++) {
				*getDataMatrixPointer(&matrix, x, y) += getDataMatrix(&lhs, x, z) * getDataMatrix(&rhs, z, y);
			}
		}
	}

	return matrix;
}

__host__ __device__
Tuple operator*(Matrix lhs, Tuple rhs) {
	return {
		((getDataMatrix(&lhs, 0, 0) * rhs.x) + (getDataMatrix(&lhs, 0, 1) * rhs.y) + (getDataMatrix(&lhs, 0, 2) * rhs.z) + (getDataMatrix(&lhs, 0, 3) * rhs.w)),
 		((getDataMatrix(&lhs, 1, 0) * rhs.x) + (getDataMatrix(&lhs, 1, 1) * rhs.y) + (getDataMatrix(&lhs, 1, 2) * rhs.z) + (getDataMatrix(&lhs, 1, 3) * rhs.w)),
 		((getDataMatrix(&lhs, 2, 0) * rhs.x) + (getDataMatrix(&lhs, 2, 1) * rhs.y) + (getDataMatrix(&lhs, 2, 2) * rhs.z) + (getDataMatrix(&lhs, 2, 3) * rhs.w)),
 		((getDataMatrix(&lhs, 3, 0) * rhs.x) + (getDataMatrix(&lhs, 3, 1) * rhs.y) + (getDataMatrix(&lhs, 3, 2) * rhs.z) + (getDataMatrix(&lhs, 3, 3) * rhs.w))
	};
}

float cofactor(Matrix matrix, int row, int column);

float determinate(Matrix matrix) {
	if (matrix.rowCount == 2 && matrix.columnCount == 2) {
		return ((getDataMatrix(&matrix, 0, 0) * getDataMatrix(&matrix, 1, 1)) - (getDataMatrix(&matrix, 1, 0) * getDataMatrix(&matrix, 0, 1)));
	}
	
	float det = 0;
	for (int y = 0; y < matrix.columnCount; y++) {
		det += getDataMatrix(&matrix, 0, y) * cofactor(matrix, 0, y);
	}

	return det;
}

Matrix submatrix(Matrix matrix, int row, int column) {
	Matrix sub = createMatrix(matrix.rowCount - 1, matrix.columnCount - 1);

	int currentX = 0;
	int currentY = 0;
	for (int x = 0; x < matrix.rowCount; x++) {
		for (int y = 0; y < matrix.columnCount; y++) {
			if (x != row && y != column) {
				setDataMatrix(&sub, currentX, currentY, getDataMatrix(&matrix, x, y));
				if (currentY + 1 == sub.columnCount) {
					currentX += 1;
					currentY = 0;
				}
				else {
					currentY += 1;
				}
			}
		}
	}

	return sub;
}

float matrixMinor(Matrix matrix, int row, int column) {
	Matrix minor = submatrix(matrix, row, column);
	return determinate(minor);
}

float cofactor(Matrix matrix, int row, int column) {
	if ((row + column) % 2 == 0) {
		return matrixMinor(matrix, row, column);
	}

	return -matrixMinor(matrix, row, column);
}

Matrix inverse(Matrix matrix) {
	Matrix inverseMatrix = createMatrix(matrix.rowCount, matrix.columnCount);
	float det = determinate(matrix);

	for (int x = 0; x < inverseMatrix.rowCount; x++) {
		for (int y = 0; y < inverseMatrix.columnCount; y++) {
			float c = cofactor(matrix, x, y);

			setDataMatrix(&inverseMatrix, y, x, c / det);
		}
	}

	return inverseMatrix;
}

__device__
Tuple multiplyFlatMatrixTuple(float* matrix, Tuple tuple) {
	return {
		((matrix[0] * tuple.x) + (matrix[1] * tuple.y) + (matrix[2] * tuple.z) + (matrix[3] * tuple.w)),
 		((matrix[4] * tuple.x) + (matrix[5] * tuple.y) + (matrix[6] * tuple.z) + (matrix[7] * tuple.w)),
 		((matrix[8] * tuple.x) + (matrix[9] * tuple.y) + (matrix[10] * tuple.z) + (matrix[11] * tuple.w)),
 		((matrix[12] * tuple.x) + (matrix[13] * tuple.y) + (matrix[14] * tuple.z) + (matrix[15] * tuple.w))
	};
}