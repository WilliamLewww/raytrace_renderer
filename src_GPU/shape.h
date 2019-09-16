#pragma once
#include "matrix.h"

enum SHAPES { SHAPES_SPHERE, SHAPES_PLANE };

struct Shape {
	SHAPES shape;

	Tuple origin;
	Matrix modelMatrix;
};

Shape createSphere() {
	return { SHAPES_SPHERE, { 0, 0, 0, 1 }, createIdentityMatrix(4) };
}

__device__
int intersectCount(Shape shape, Ray ray, float* modelMatrixData) {
	int intersectionCount = 0;

	float* inverseModelMatrix = inverseFlat(modelMatrixData);

	return intersectionCount;
}