#pragma once
#include "tuple.h"
#include "matrix.h"

struct Ray {
	Tuple origin;
	Tuple direction;
};

__device__ Ray transform(Ray ray, float* matrix);