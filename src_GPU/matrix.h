#pragma once

struct Matrix {
	float* data;

	int rowCount;
	int columnCount;
};

__host__ __device__
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