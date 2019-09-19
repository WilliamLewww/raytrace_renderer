#pragma once
#include "matrix.h"
#include "ray.h"

enum SHAPES { SHAPES_SPHERE, SHAPES_PLANE };

struct Shape {
	SHAPES shape;

	Tuple origin;
	Matrix modelMatrix;
};

Shape createSphere();

__device__
int intersectCount(Shape shape, Ray ray, float* modelMatrixData);