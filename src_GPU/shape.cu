#include "shape.h"
#include <stdio.h>

Shape createSphere() {
	return { SHAPES_SPHERE, { 0, 0, 0, 1 }, createIdentityMatrix(4) };
}

__device__
int intersectSphereCount(Shape shape, Ray ray) {
	Tuple sphereToRay = ray.origin - shape.origin;
	float a = dot(ray.direction, ray.direction);
	float b = 2 * dot(ray.direction, sphereToRay);
	float c = dot(sphereToRay, sphereToRay) - 1;

	float discriminant = pow(b, 2) - (4 * a * c);

	if (discriminant < 0) {
		return 0;
	}

	if ((-b - sqrt(discriminant)) / (2 * a) == (-b + sqrt(discriminant)) / (2 * a)) {
		return 1;
	}

	return 2;
}

__device__
int intersectCount(Shape shape, Ray ray, float* inverseModelMatrixData) {
	int intersectionCount = 0;

	Ray rayTransformed = transform(ray, inverseModelMatrixData);
	if (shape.shape == SHAPES_SPHERE) {
		intersectionCount = intersectSphereCount(shape, rayTransformed);
	}
	
	return intersectionCount;
}