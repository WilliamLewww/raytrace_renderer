#pragma once

struct Matrix {
	float** data;

	int rowCount;
	int columnCount;

	float* operator[](int x) {
		return data[x];
	};
};

const float getElementFromMatrix(Matrix matrix, int x, int y) {
	return matrix[x][y];
}

bool operator==(Matrix matrixA, Matrix matrixB) {
	bool greater = false;

	if (matrixA.rowCount != matrixB.rowCount || matrixA.columnCount != matrixB.columnCount) {
		return false;
	}

	for (int x = 0; !greater && x < matrixA.rowCount; x++) {
		for (int y = 0; !greater && y < matrixA.columnCount; y++) {
			if (abs(matrixA[x][y] - matrixB[x][y]) >= EPSILON_COMPARISON) {
				greater = true;
			}
		}
	}

	return !greater;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
	for (int x = 0; x < matrix.rowCount; x++) {
		os << "[";
		for (int y = 0; y < matrix.columnCount; y++) {
			os << getElementFromMatrix(matrix, x, y);
			if (y < matrix.columnCount - 1) { os << ", "; }
		}
		os << "]";
		if (x < matrix.rowCount - 1) { os << "\n"; }
		
	}

	return os;
}

Matrix createMatrix(int rowCount, int columnCount) {
	Matrix temp;
	temp.data = new float*[rowCount];
	for (int x = 0; x < rowCount; x++) {
		temp.data[x] = new float[columnCount];
		for (int y = 0; y < columnCount; y++) {
			temp.data[x][y] = 0.0;
		}
	}

	temp.rowCount = rowCount;
	temp.columnCount = columnCount;

	return temp;
}

Matrix createIdentityMatrix(int rowCount) {
	Matrix temp = createMatrix(rowCount, rowCount);
	for (int x = 0; x < rowCount; x++) {
		temp[x][x] = 1.0;
	}

	return temp;
}

Matrix operator*(Matrix matrixA, Matrix matrixB) {
	Matrix temp = createMatrix(matrixA.rowCount, matrixB.columnCount);
	for (int x = 0; x < matrixA.rowCount; x++) {
		for (int y = 0; y < matrixB.columnCount; y++) {
			for (int z = 0; z < matrixA.columnCount; z++) {
				temp[x][y] += (matrixA[x][z] * matrixB[z][y]);
			}
		}
	}

	return temp;
}

Tuple operator*(Matrix matrix, Tuple tuple) {
	return {
		((matrix[0][0] * tuple.x) + (matrix[0][1] * tuple.y) + (matrix[0][2] * tuple.z) + (matrix[0][3] * tuple.w)),
 		((matrix[1][0] * tuple.x) + (matrix[1][1] * tuple.y) + (matrix[1][2] * tuple.z) + (matrix[1][3] * tuple.w)),
 		((matrix[2][0] * tuple.x) + (matrix[2][1] * tuple.y) + (matrix[2][2] * tuple.z) + (matrix[2][3] * tuple.w)),
 		((matrix[3][0] * tuple.x) + (matrix[3][1] * tuple.y) + (matrix[3][2] * tuple.z) + (matrix[3][3] * tuple.w))
	};
}

Matrix transpose(Matrix matrix) {
	Matrix temp = createMatrix(matrix.columnCount, matrix.rowCount);
	for (int x = 0; x < matrix.rowCount; x++) {
		for (int y = 0; y < matrix.columnCount; y++) {
			temp[y][x] = matrix[x][y];
		}
	}

	return temp;
}

float cofactor(Matrix matrix, int row, int column);
float determinate(Matrix matrix) {
	if (matrix.rowCount == 2 && matrix.columnCount == 2) {
		return ((matrix[0][0] * matrix[1][1]) - (matrix[1][0] * matrix[0][1]));
	}
	
	float det = 0;
	for (int y = 0; y < matrix.columnCount; y++) {
		det += matrix[0][y] * cofactor(matrix, 0, y);
	}

	return det;
}

Matrix submatrix(Matrix matrix, int row, int column) {
	Matrix temp = createMatrix(matrix.rowCount - 1, matrix.columnCount - 1);

	int currentX = 0;
	int currentY = 0;
	for (int x = 0; x < matrix.rowCount; x++) {
		for (int y = 0; y < matrix.columnCount; y++) {
			if (x != row && y != column) {
				temp[currentX][currentY] = matrix[x][y];
				if (currentY + 1 == temp.columnCount) {
					currentX += 1;
					currentY = 0;
				}
				else {
					currentY += 1;
				}
			}
		}
	}

	return temp;
}

float matrixMinor(Matrix matrix, int row, int column) {
	Matrix temp = submatrix(matrix, row, column);
	return determinate(temp);
}

float cofactor(Matrix matrix, int row, int column) {
	if ((row + column) % 2 == 0) {
		return matrixMinor(matrix, row, column);
	}

	return -matrixMinor(matrix, row, column);
}

Matrix inverse(Matrix matrix) {
	Matrix temp = createMatrix(matrix.rowCount, matrix.columnCount);
	float det = determinate(matrix);

	for (int x = 0; x < temp.rowCount; x++) {
		for (int y = 0; y < temp.columnCount; y++) {
			float c = cofactor(matrix, x, y);

			temp[y][x] = c / det;
		}
	}

	return temp;
}