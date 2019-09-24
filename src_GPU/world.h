#pragma once
#include "light.h"
#include "shape.h"

struct World {
	PointLight* lightArray;
	Shape* shapeArray;

	int pointLightCount;
	int shapeCount;
};

World createDefaultWorld();
void colorAt(Tuple* colorOut, World world, Ray* rays, int rayCount);

__device__ int intersectWorldCount(Shape* shapes, int shapeCount, Ray ray, float* allModelMatrixData);
__global__ void colorAtKernel(Tuple* colorBuffer, Ray* rays, int rayCount, Shape* shapes, int shapeCount, float* allModelMatrixData);