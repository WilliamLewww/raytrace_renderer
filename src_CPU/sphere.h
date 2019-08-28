#pragma once
#include "matrix.h"
#include "ray.h"
#include "light.h"

struct Shape {
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

Shape createShape() {
	return { createPoint(), createIdentityMatrix(4) };
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

Intersection* intersect(Shape& shape, Ray ray, int& intersectionCount) {
	Ray rayTransformed = transform(ray, inverse(shape.modelMatrix));

	Tuple sphereToRay = rayTransformed.origin - shape.origin;
	float a = dot(rayTransformed.direction, rayTransformed.direction);
	float b = 2 * dot(rayTransformed.direction, sphereToRay);
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

Tuple normalAt(Shape shape, Tuple point) {
	Tuple objectToPoint = inverse(shape.modelMatrix) * point;
	Tuple objectNormal = objectToPoint - shape.origin;
	Tuple worldNormal = transpose(inverse(shape.modelMatrix)) * objectNormal;
	worldNormal.w = 0;

	return normalize(worldNormal);
}