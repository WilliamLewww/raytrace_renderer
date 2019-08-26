#pragma once
#include "tuple.h"

struct Ray {
	Tuple origin;
	Tuple direction;
};

struct Sphere {
	Tuple origin;
	float radius;
};

struct Intersection {
	float t;
	Sphere* object;
};

std::ostream& operator<<(std::ostream& os, const Ray& ray) {
    os << "origin: " << ray.origin << ", direction: " << ray.direction;
    return os;
}

std::ostream& operator<<(std::ostream& os, const Intersection& intersection) {
    os << "t: " << intersection.t << ", object: " << intersection.object;
    return os;
}

Ray createRay(Tuple origin, Tuple direction) {
	return { origin, direction };
}

Sphere createSphere() {
	return { createPoint(), 1.0 };
}

Intersection createIntersection(float t, Sphere* object) {
	return { t, object };
}

Tuple project(Ray ray, float t) {
	return ray.origin + (ray.direction * t);
}

Intersection* intersect(Sphere& sphere, Ray ray, int& intersectionCount) {
	Tuple sphereToRay = ray.origin - sphere.origin;
	float a = dot(ray.direction, ray.direction);
	float b = 2 * dot(ray.direction, sphereToRay);
	float c = dot(sphereToRay, sphereToRay) - 1;

	float discriminant = pow(b, 2) - (4 * a * c);

	if (discriminant < 0) {
		intersectionCount = 0;

		return nullptr;
	}

	if ((-b - sqrt(discriminant)) / (2 * a) == (-b + sqrt(discriminant)) / (2 * a)) {
		intersectionCount = 1;

		Intersection* intersection = new Intersection[1];
		intersection[0] = createIntersection((-b - sqrt(discriminant)) / (2 * a), &sphere);

		return intersection;
	}

	intersectionCount = 2;

	Intersection* intersection = new Intersection[2];
	intersection[0] = createIntersection((-b - sqrt(discriminant)) / (2 * a), &sphere);
	intersection[1] = createIntersection((-b + sqrt(discriminant)) / (2 * a), &sphere);
	return intersection;
}