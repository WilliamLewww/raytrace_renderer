#include "shape.h"
#include <stdio.h>

Shape createSphere() {
	return { SHAPES_SPHERE, { 0, 0, 0, 1 }, createIdentityMatrix(4) };
}

__device__
int intersectCount(Shape shape, Ray ray, float* inverseModelMatrixData) {
	int intersectionCount = 0;

	Ray rayTransformed = transform(ray, inverseModelMatrixData);
	
	return intersectionCount;
}