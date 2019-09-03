#pragma once
#include <algorithm>
#include "shape.h"
#include "transform.h"
#include "constants.h"

struct Precomputed {
	float t;
	Shape* object;

	Tuple point;
	Tuple eyeV;
	Tuple normalV;

	Tuple overPoint;

	bool inside;
};

struct World {
	PointLight* lightArray;
	Shape* shapeArray;

	int pointLightCount;
	int shapeCount;
};

World createDefaultWorld() {
	World world;

	world.pointLightCount = 1;
	world.shapeCount = 2;

	world.lightArray = new PointLight[1];
	world.lightArray[0] = createPointLight(createPoint(-10, 10, -10), createColor(1, 1, 1));

	world.shapeArray = new Shape[2];
	world.shapeArray[0] = createSphere();
	world.shapeArray[0].material = createMaterial();
	world.shapeArray[0].material.color = createColor(0.8, 1.0, 0.6);
	world.shapeArray[0].material.diffuse = 0.7;
	world.shapeArray[0].material.specular = 0.2;
	world.shapeArray[0].modelMatrix = createIdentityMatrix(4);

	world.shapeArray[1] = createSphere();
	world.shapeArray[1].material = createMaterial();
	world.shapeArray[1].modelMatrix = createScaleMatrix(0.5, 0.5, 0.5);

	return world;
}

World createWorld1() {
	World world;

	world.pointLightCount = 1;
	world.shapeCount = 6;

	world.lightArray = new PointLight[world.pointLightCount];
	world.lightArray[0] = createPointLight(createPoint(-10, 10, -10), createColor(1, 1, 1));

	world.shapeArray = new Shape[world.shapeCount];
	world.shapeArray[0] = createSphere();
	world.shapeArray[0].material = createMaterial();
	world.shapeArray[0].material.color = createColor(1.0, 0.9, 0.9);
	world.shapeArray[0].material.specular = 0;
	world.shapeArray[0].modelMatrix = createScaleMatrix(10, 0.01, 10);

	world.shapeArray[1] = createSphere();
	world.shapeArray[1].material = createMaterial();
	world.shapeArray[1].material.color = createColor(1.0, 0.9, 0.9);
	world.shapeArray[1].material.specular = 0;
	world.shapeArray[1].modelMatrix = createTranslateMatrix(0, 0, 5) * createRotationMatrixY(-M_PI / 4) * createRotationMatrixX(M_PI / 2) * createScaleMatrix(10, 0.01, 10);

	world.shapeArray[2] = createSphere();
	world.shapeArray[2].material = createMaterial();
	world.shapeArray[2].material.color = createColor(1.0, 0.9, 0.9);
	world.shapeArray[2].material.specular = 0;
	world.shapeArray[2].modelMatrix = createTranslateMatrix(0, 0, 5) * createRotationMatrixY(M_PI / 4) * createRotationMatrixX(M_PI / 2) * createScaleMatrix(10, 0.01, 10);

	world.shapeArray[3] = createSphere();
	world.shapeArray[3].material = createMaterial();
	world.shapeArray[3].material.color = createColor(0.1, 1.0, 0.5);
	world.shapeArray[3].material.diffuse = 0.7;
	world.shapeArray[3].material.specular = 0.3;
	world.shapeArray[3].modelMatrix = createTranslateMatrix(-0.5, 1, 0.5);

	world.shapeArray[4] = createSphere();
	world.shapeArray[4].material = createMaterial();
	world.shapeArray[4].material.color = createColor(0.5, 1.0, 0.1);
	world.shapeArray[4].material.diffuse = 0.7;
	world.shapeArray[4].material.specular = 0.3;
	world.shapeArray[4].modelMatrix = createTranslateMatrix(1.5, 0.5, -0.5) * createScaleMatrix(0.5, 0.5, 0.5);

	world.shapeArray[5] = createSphere();
	world.shapeArray[5].material = createMaterial();
	world.shapeArray[5].material.color = createColor(1.0, 0.8, 0.1);
	world.shapeArray[5].material.diffuse = 0.7;
	world.shapeArray[5].material.specular = 0.3;
	world.shapeArray[5].modelMatrix = createTranslateMatrix(-1.5, 0.3, -0.7) * createScaleMatrix(0.3, 0.3, 0.3);

	return world;
}

World createWorld2() {
	World world;

	world.pointLightCount = 1;
	world.shapeCount = 8;

	world.lightArray = new PointLight[world.pointLightCount];
	world.lightArray[0] = createPointLight(createPoint(10, 10, -3), createColor(1, 1, 1));

	world.shapeArray = new Shape[world.shapeCount];

	world.shapeArray[0] = createPlane();
	world.shapeArray[0].material = createMaterial();
	world.shapeArray[0].material.color = createColor(0.9, 0.5, 0.9);
	world.shapeArray[0].material.specular = 0;
	world.shapeArray[0].modelMatrix = createTranslateMatrix(0, 0, 3) * createRotationMatrixX(-M_PI / 4);

	world.shapeArray[1] = createPlane();
	world.shapeArray[1].material = createMaterial();
	world.shapeArray[1].material.color = createColor(0.9, 0.9, 0.5);
	world.shapeArray[1].material.specular = 0;
	world.shapeArray[1].modelMatrix = createTranslateMatrix(-3, 0, 0) * createRotationMatrixZ(-M_PI / 4);

	world.shapeArray[2] = createPlane();
	world.shapeArray[2].material = createMaterial();
	world.shapeArray[2].material.color = createColor(5.0, 0.9, 0.9);
	world.shapeArray[2].material.specular = 0;
	world.shapeArray[2].modelMatrix = createIdentityMatrix(4);

	world.shapeArray[3] = createSphere();
	world.shapeArray[3].material = createMaterial();
	world.shapeArray[3].material.color = createColor(0.7, 1.0, 0.2);
	world.shapeArray[3].material.diffuse = 0.7;
	world.shapeArray[3].material.specular = 0.3;
	world.shapeArray[3].modelMatrix = createTranslateMatrix(-0.5, 1, 0.5);

	world.shapeArray[4] = createSphere();
	world.shapeArray[4].material = createMaterial();
	world.shapeArray[4].material.color = createColor(1.0, 0.5, 0.5);
	world.shapeArray[4].material.diffuse = 0.7;
	world.shapeArray[4].material.specular = 0.3;
	world.shapeArray[4].modelMatrix = createTranslateMatrix(-0.5, 2, 0.5) * createScaleMatrix(0.75, 0.75, 0.75);

	world.shapeArray[5] = createSphere();
	world.shapeArray[5].material = createMaterial();
	world.shapeArray[5].material.color = createColor(0.3, 0.2, 0.5);
	world.shapeArray[5].material.diffuse = 0.7;
	world.shapeArray[5].material.specular = 0.3;
	world.shapeArray[5].modelMatrix = createTranslateMatrix(2, 1, 0.5) * createScaleMatrix(1.25, 1.25, 1.25);

	world.shapeArray[6] = createSphere();
	world.shapeArray[6].material = createMaterial();
	world.shapeArray[6].material.color = createColor(1.0, 1.0, 1.0);
	world.shapeArray[6].material.diffuse = 0.7;
	world.shapeArray[6].material.specular = 0.3;
	world.shapeArray[6].modelMatrix = createTranslateMatrix(-0.5, 0.25, -1.5) * createScaleMatrix(0.5, 0.5, 0.5);

	world.shapeArray[7] = createSphere();
	world.shapeArray[7].material = createMaterial();
	world.shapeArray[7].material.color = createColor(0.3, 0.3, 1.0);
	world.shapeArray[7].material.diffuse = 0.7;
	world.shapeArray[7].material.specular = 0.3;
	world.shapeArray[7].modelMatrix = createTranslateMatrix(-0.5, 2, -2.0) * createScaleMatrix(0.25, 0.25, 0.25);

	return world;
}

#include <vector>

bool sortIntersections(Intersection intersectionA, Intersection intersectionB) {
    return intersectionA.t >= intersectionB.t;
}

Intersection* intersectWorld(World world, Ray ray, int& intersectionCount) {
	int tempIntersectionCount = 0;
	std::vector<Intersection> intersections;
	for (int x = 0; x < world.shapeCount; x++) {
		Intersection* tempIntersections = intersect(world.shapeArray[x], ray, tempIntersectionCount);

		for (int y = 0; y < tempIntersectionCount; y++) {
			intersections.push_back(tempIntersections[y]);
		}
	}

	std::sort(intersections.begin(), intersections.end(), sortIntersections);

	intersectionCount = intersections.size();
	return intersections.data();
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

	return lighting(precomputed.object->material, world.lightArray[0], precomputed.overPoint, precomputed.eyeV, precomputed.normalV, inShadow);
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