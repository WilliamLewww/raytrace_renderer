#pragma once

//FIX MEMORY LEAK

struct Matrix4 {
	float** data;

	float* operator[](int x) {
		return data[x];
	};
};

bool operator==(Matrix4 matrixA, Matrix4 matrixB) {
	bool greater = false;

	for (int x = 0; !greater && x < 4; x++) {
		for (int y = 0; !greater && y < 4; y++) {
			if (abs(matrixA[x][y] - matrixB[x][y]) >= EPSILON_TUPLE) {
				greater = true;
			}
		}
	}

	return !greater;
}

std::ostream& operator<<(std::ostream& os, Matrix4& matrix) {
	for (int x = 0; x < 4; x++) {
		os << "[";
		for (int y = 0; y < 4; y++) {
			os << matrix[x][y];
			if (y < 3) { os << ", "; }
		}
		os << "]";
		if (x < 3) { os << "\n"; }
		
	}

	return os;
}

Matrix4 createMatrix4() {
	Matrix4 temp;
	temp.data = new float*[4];
	for (int x = 0; x < 4; x++) {
		temp.data[x] = new float[4];
		for (int y = 0; y < 4; y++) {
			temp.data[x][y] = 0.0;
		}
	}

	return temp;
}

Matrix4 operator*(Matrix4 matrixA, Matrix4 matrixB) {
	Matrix4 temp = createMatrix4();
}