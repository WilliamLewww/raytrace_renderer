#pragma once
#include "matrix.h"

Matrix createTranslateMatrix(float x, float y, float z) {
	Matrix matrix = createIdentityMatrix(4);

	setDataMatrix(&matrix, 0, 3, x);
	setDataMatrix(&matrix, 1, 3, y);
	setDataMatrix(&matrix, 2, 3, z);

	return matrix;
}