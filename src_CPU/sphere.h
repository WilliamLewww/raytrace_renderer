#pragma once
#include "matrix.h"
#include "ray.h"
#include "light.h"

enum SHAPES { SHAPES_SPHERE, SHAPES_PLANE };

struct Shape {
	SHAPES shape;

	Tuple origin;

	Matrix modelMatrix;
	Material material;
};

struct Intersection {
	float t;
	Shape* object;
};

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << "origin: " << shape.origin;
    return os;
}

std::ostream& operator<<(std::ostream& os, const Intersection& intersection) {
    os << "t: " << intersection.t << ", object: " << intersection.object;
    return os;
}

Shape createSphere() {
	return { SHAPES_SPHERE, createPoint(), createIdentityMatrix(4) };
}

Shape createPlane() {
	return { SHAPES_PLANE, createPoint(), createIdentityMatrix(4) };
}

Intersection createIntersection(float t, Shape* object) {
	return { t, object };
}

Intersection* hit(Intersection* intersections, int intersectionCount) {
	Intersection* closestHit = nullptr;
	for (int x = 0; x < intersectionCount; x++) {
		if (closestHit == nullptr && intersections[x].t > 0) {
			closestHit = &intersections[x];
		}
		else {
			if (closestHit != nullptr && intersections[x].t < closestHit->t && intersections[x].t > 0) {
				closestHit = &intersections[x];
			}
		}
	}

	return closestHit;
}

Intersection* intersectSphere(Shape& shape, Ray ray, int& intersectionCount) {
	Tuple sphereToRay = ray.origin - shape.origin;
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
		intersection[0] = createIntersection((-b - sqrt(discriminant)) / (2 * a), &shape);

		return intersection;
	}

	intersectionCount = 2;

	Intersection* intersection = new Intersection[2];
	intersection[0] = createIntersection((-b - sqrt(discriminant)) / (2 * a), &shape);
	intersection[1] = createIntersection((-b + sqrt(discriminant)) / (2 * a), &shape);
	return intersection;
}

Intersection* intersectPlane(Shape& shape, Ray ray, int& intersectionCount) {
	if (fabs(ray.direction.y) < EPSILON_COMPARISON) {
		intersectionCount = 0;
		return nullptr;
	}

	intersectionCount = 1;
	Intersection* intersection = new Intersection[1];
	intersection[0] = createIntersection(abs(ray.origin.y / ray.direction.y), &shape);

	return intersection;
}

Tuple normalAtSphere(Shape shape, Tuple point) {
	return inverse(shape.modelMatrix) * point - shape.origin;
}

Tuple normalAtPlane(Shape shape, Tuple point) {
	return createVector(0, 1, 0);
}

Intersection* intersect(Shape& shape, Ray ray, int& intersectionCount) {
	Intersection* intersections;

	Ray rayTransformed = transform(ray, inverse(shape.modelMatrix));
	if (shape.shape == SHAPES_SPHERE) {
		intersections = intersectSphere(shape, rayTransformed, intersectionCount);
	}
	if (shape.shape == SHAPES_PLANE) {
		intersections = intersectPlane(shape, rayTransformed, intersectionCount);
	}

	return intersections;
}

Tuple normalAt(Shape shape, Tuple point) {
	Tuple objectToPoint = inverse(shape.modelMatrix) * point;

	Tuple objectNormal;
	if (shape.shape == SHAPES_SPHERE) {
		objectNormal = normalAtSphere(shape, point);
	}
	if (shape.shape == SHAPES_PLANE) {
		objectNormal = normalAtPlane(shape, point);
	}

	Tuple worldNormal = transpose(inverse(shape.modelMatrix)) * objectNormal;
	worldNormal.w = 0;

	return normalize(worldNormal);
}