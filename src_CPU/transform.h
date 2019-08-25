#pragma once
#include <cmath>
#include "matrix.h"

Matrix createTranslateMatrix(float x, float y, float z) {
	Matrix temp = createIdentityMatrix(4);
	temp[0][3] = x;
	temp[1][3] = y;
	temp[2][3] = z;

	return temp;
}

Matrix createScaleMatrix(int x, int y, int z) {
	Matrix temp = createIdentityMatrix(4);
	temp[0][0] = x;
	temp[1][1] = y;
	temp[2][2] = z;

	return temp;
}

Matrix createRotationMatrixX(float radians) {
	Matrix temp = createIdentityMatrix(4);
	temp[1][1] = cos(radians);
	temp[1][2] = -sin(radians);
	temp[2][1] = sin(radians);
	temp[2][2] = cos(radians);

	return temp;
}

Matrix createRotationMatrixY(float radians) {
	Matrix temp = createIdentityMatrix(4);
	temp[0][0] = cos(radians);
	temp[0][2] = sin(radians);
	temp[2][0] = -sin(radians);
	temp[2][2] = cos(radians);

	return temp;
}

Matrix createRotationMatrixZ(float radians) {
	Matrix temp = createIdentityMatrix(4);
	temp[0][0] = cos(radians);
	temp[0][1] = -sin(radians);
	temp[1][0] = sin(radians);
	temp[1][1] = cos(radians);

	return temp;
}

Matrix createShearMatrix(float xy, float xz, float yx, float yz, float zx, float zy) {
	Matrix temp = createIdentityMatrix(4);
	temp[0][1] = xy;
	temp[0][2] = xz;
	temp[1][0] = yx;
	temp[1][2] = yz;
	temp[2][0] = zx;
	temp[2][1] = zy;

	return temp;
}