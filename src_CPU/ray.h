#pragma once
#include "tuple.h"
#include "matrix.h"

struct Ray {
	Tuple origin;
	Tuple direction;
};

std::ostream& operator<<(std::ostream& os, const Ray& ray) {
    os << "origin: " << ray.origin << ", direction: " << ray.direction;
    return os;
}

Ray createRay(Tuple origin, Tuple direction) {
	return { origin, direction };
}

Tuple project(Ray ray, float t) {
	return ray.origin + (ray.direction * t);
}

Ray transform(Ray ray, Matrix matrix) {
	return createRay(matrix * ray.origin, matrix * ray.direction);
}