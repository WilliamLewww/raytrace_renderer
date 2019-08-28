#pragma once
#include <algorithm>
#include "sphere.h"
#include "light.h"

const float EPSILON_NOISE = 0.01;

struct Precomputed {
	float t;
	Sphere* object;

	Tuple point;
	Tuple eyeV;
	Tuple normalV;

	Tuple overPoint;

	bool inside;
};

struct World {
	PointLight* lightArray;
	Sphere* sphereArray;

	int pointLightCount;
	int sphereCount;
};

World createDefaultWorld() {
	World world;

	world.pointLightCount = 1;
	world.sphereCount = 2;

	world.lightArray = new PointLight[1];
	world.lightArray[0] = createPointLight(createPoint(-10, 10, -10), createColor(1, 1, 1));

	world.sphereArray = new Sphere[2];
	world.sphereArray[0] = createSphere();
	world.sphereArray[0].shape.material = createMaterial();
	world.sphereArray[0].shape.material.color = createColor(0.8, 1.0, 0.6);
	world.sphereArray[0].shape.material.diffuse = 0.7;
	world.sphereArray[0].shape.material.specular = 0.2;
	world.sphereArray[0].shape.modelMatrix = createIdentityMatrix(4);

	world.sphereArray[1] = createSphere();
	world.sphereArray[1].shape.material = createMaterial();
	world.sphereArray[1].shape.modelMatrix = createScaleMatrix(0.5, 0.5, 0.5);

	return world;
}

World createWorld1() {
	World world;

	world.pointLightCount = 1;
	world.sphereCount = 6;

	world.lightArray = new PointLight[world.pointLightCount];
	world.lightArray[0] = createPointLight(createPoint(-10, 10, -10), createColor(1, 1, 1));

	world.sphereArray = new Sphere[world.sphereCount];
	world.sphereArray[0] = createSphere();
	world.sphereArray[0].shape.material = createMaterial();
	world.sphereArray[0].shape.material.color = createColor(1.0, 0.9, 0.9);
	world.sphereArray[0].shape.material.specular = 0;
	world.sphereArray[0].shape.modelMatrix = createScaleMatrix(10, 0.01, 10);

	world.sphereArray[1] = createSphere();
	world.sphereArray[1].shape.material = createMaterial();
	world.sphereArray[1].shape.material.color = createColor(1.0, 0.9, 0.9);
	world.sphereArray[1].shape.material.specular = 0;
	world.sphereArray[1].shape.modelMatrix = createTranslateMatrix(0, 0, 5) * createRotationMatrixY(-M_PI / 4) * createRotationMatrixX(M_PI / 2) * createScaleMatrix(10, 0.01, 10);

	world.sphereArray[2] = createSphere();
	world.sphereArray[2].shape.material = createMaterial();
	world.sphereArray[2].shape.material.color = createColor(1.0, 0.9, 0.9);
	world.sphereArray[2].shape.material.specular = 0;
	world.sphereArray[2].shape.modelMatrix = createTranslateMatrix(0, 0, 5) * createRotationMatrixY(M_PI / 4) * createRotationMatrixX(M_PI / 2) * createScaleMatrix(10, 0.01, 10);

	world.sphereArray[3] = createSphere();
	world.sphereArray[3].shape.material = createMaterial();
	world.sphereArray[3].shape.material.color = createColor(0.1, 1.0, 0.5);
	world.sphereArray[3].shape.material.diffuse = 0.7;
	world.sphereArray[3].shape.material.specular = 0.3;
	world.sphereArray[3].shape.modelMatrix = createTranslateMatrix(-0.5, 1, 0.5);

	world.sphereArray[4] = createSphere();
	world.sphereArray[4].shape.material = createMaterial();
	world.sphereArray[4].shape.material.color = createColor(0.5, 1.0, 0.1);
	world.sphereArray[4].shape.material.diffuse = 0.7;
	world.sphereArray[4].shape.material.specular = 0.3;
	world.sphereArray[4].shape.modelMatrix = createTranslateMatrix(1.5, 0.5, -0.5) * createScaleMatrix(0.5, 0.5, 0.5);

	world.sphereArray[5] = createSphere();
	world.sphereArray[5].shape.material = createMaterial();
	world.sphereArray[5].shape.material.color = createColor(1.0, 0.8, 0.1);
	world.sphereArray[5].shape.material.diffuse = 0.7;
	world.sphereArray[5].shape.material.specular = 0.3;
	world.sphereArray[5].shape.modelMatrix = createTranslateMatrix(-1.5, 0.3, -0.7) * createScaleMatrix(0.3, 0.3, 0.3);

	return world;
}

bool sortIntersections(Intersection intersectionA, Intersection intersectionB) {
    return intersectionA.t < intersectionB.t;
}

Intersection* intersectWorld(World world, Ray ray, int& intersectionCount) {
	int totalIntersectionCount = 0;
	int tempIntersectionCount;
	for (int x = 0; x < world.sphereCount; x++) {
		intersect(world.sphereArray[x], ray, tempIntersectionCount);
		totalIntersectionCount += tempIntersectionCount;
	}
	intersectionCount = totalIntersectionCount;

	Intersection* intersection = new Intersection[totalIntersectionCount];
	int currentIntersection = 0;
	for (int x = 0; x < world.sphereCount; x++) {
		Intersection* tempIntersections = intersect(world.sphereArray[x], ray, tempIntersectionCount);
		if (tempIntersectionCount > 0) {
			for (int y = 0; y < tempIntersectionCount; y++) {
				intersection[currentIntersection] = tempIntersections[y];
				currentIntersection += 1;
			}
		}
	}

	std::sort(intersection, intersection + totalIntersectionCount, &sortIntersections);
	return intersection;
}

Precomputed prepareComputations(Intersection intersection, Ray ray) {
	Precomputed precomputed;

	precomputed.t = intersection.t;
	precomputed.object = intersection.object;

	precomputed.point = project(ray, precomputed.t);
	precomputed.eyeV = negate(ray.direction);
	precomputed.normalV = normalAt(*precomputed.object, precomputed.point);

	if (dot(precomputed.normalV, precomputed.eyeV) < 0) {
		precomputed.inside = true;
		precomputed.normalV = negate(precomputed.normalV);
	}
	else {
		precomputed.inside = false;
	}

	precomputed.overPoint = precomputed.point + precomputed.normalV * EPSILON_NOISE;

	return precomputed;
}

bool isShadowed(World world, Tuple point) {
	Tuple pointToLight = world.lightArray[0].position - point;
	float distance = magnitude(pointToLight);
	Tuple pointToLightN = normalize(pointToLight);

	Ray ray = createRay(point, pointToLightN);

	int intersectionCount;
	Intersection* intersections = intersectWorld(world, ray, intersectionCount);

	Intersection* closestHit = hit(intersections, intersectionCount);
	if (closestHit != nullptr && closestHit->t < distance) {
		return true;
	}

	return false;
}

Tuple shadeHit(World world, Precomputed precomputed) {
	bool inShadow = isShadowed(world, precomputed.overPoint);

	return lighting(precomputed.object->shape.material, world.lightArray[0], precomputed.overPoint, precomputed.eyeV, precomputed.normalV, inShadow);
}

Tuple colorAt(World world, Ray ray) {
	int intersectionCount;
	Intersection* intersections = intersectWorld(world, ray, intersectionCount);

	if (intersectionCount > 0) {
		Intersection* closestHit = hit(intersections, intersectionCount);
		Precomputed computation = prepareComputations(*closestHit, ray);
		return shadeHit(world, computation);
	}

	return createColor(0, 0, 0);
}