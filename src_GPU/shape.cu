#include "shape.h"
#include <cublas_v2.h>

Shape createSphere() {
	return { SHAPES_SPHERE, { 0, 0, 0, 1 }, createIdentityMatrix(4) };
}

__device__
int intersectCount(Shape shape, Ray ray, float* modelMatrixData) {
    cublasHandle_t handle;

    cublasCreate(&handle);
    cublasDestroy(handle);

	int intersectionCount = 0;
	
	return intersectionCount;
}