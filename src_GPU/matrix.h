#pragma once

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

__host__ __device__
void setDataMatrix(Matrix* matrix, int row, int column, float value) {
	matrix->data[(row * matrix->columnCount) + column] = value;
}

__host__ __device__
float getDataMatrix(Matrix* matrix, int row, int column) {
	return matrix->data[(row * matrix->columnCount) + column];
}

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

__host__ __device__
Matrix createIdentityMatrix(int rowColumnCount) {
	Matrix matrix = createMatrix(rowColumnCount, rowColumnCount);
	for (int x = 0; x < rowColumnCount; x++) {
		setDataMatrix(&matrix, x, x, 1.0);
	}

	return matrix;
}