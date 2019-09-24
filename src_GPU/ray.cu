#include "ray.h"

__device__
Ray transform(Ray ray, float* matrix) {
	return { multiplyFlatMatrixTuple(matrix, ray.origin), multiplyFlatMatrixTuple(matrix, ray.direction) };
}