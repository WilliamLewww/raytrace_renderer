#pragma once
#include "matrix.h"
#include "ray.h"
#include "light.h"

struct Shape {
	Matrix modelMatrix;
	Material material;
};

struct Sphere {
	Tuple origin;
	float radius;

	Shape shape;
};

struct Intersection {
	float t;
	Sphere* object;
};

std::ostream& operator<<(std::ostream& os, const Sphere& sphere) {
    os << "origin: " << sphere.origin << ", radius: " << sphere.radius;
    return os;
}

std::ostream& operator<<(std::ostream& os, const Intersection& intersection) {
    os << "t: " << intersection.t << ", object: " << intersection.object;
    return os;
}

Sphere createSphere() {
	return { createPoint(), 1.0, createIdentityMatrix(4) };
}

Intersection createIntersection(float t, Sphere* object) {
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

Intersection* intersect(Sphere& sphere, Ray ray, int& intersectionCount) {
	Ray rayTransformed = transform(ray, inverse(sphere.shape.modelMatrix));

	Tuple sphereToRay = rayTransformed.origin - sphere.origin;
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
		intersection[0] = createIntersection((-b - sqrt(discriminant)) / (2 * a), &sphere);

		return intersection;
	}

	intersectionCount = 2;

	Intersection* intersection = new Intersection[2];
	intersection[0] = createIntersection((-b - sqrt(discriminant)) / (2 * a), &sphere);
	intersection[1] = createIntersection((-b + sqrt(discriminant)) / (2 * a), &sphere);
	return intersection;
}

Tuple normalAt(Sphere sphere, Tuple point) {
	Tuple objectToPoint = inverse(sphere.shape.modelMatrix) * point;
	Tuple objectNormal = objectToPoint - sphere.origin;
	Tuple worldNormal = transpose(inverse(sphere.shape.modelMatrix)) * objectNormal;
	worldNormal.w = 0;

	return normalize(worldNormal);
}