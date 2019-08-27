#pragma once
#include <algorithm>
#include "sphere.h"
#include "light.h"

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
	world.sphereArray[0].material = createMaterial();
	world.sphereArray[0].material.color = createColor(0.8, 1.0, 0.6);
	world.sphereArray[0].material.diffuse = 0.7;
	world.sphereArray[0].material.specular = 0.2;

	world.sphereArray[1] = createSphere();
	world.sphereArray[1].modelMatrix = createScaleMatrix(0.5, 0.5, 0.5);

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