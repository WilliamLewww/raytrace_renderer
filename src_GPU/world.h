#pragma once
#include "light.h"
#include "shape.h"

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
	world.lightArray[0] = {{ -10, 10, -10, 1 }, { 1, 1, 1, 1 }};

	world.shapeArray = new Shape[1];
	world.shapeArray[0] = createSphere();

	return world;
}

__device__
int intersectWorldCount(Shape* shapes, int shapeCount, Ray ray) {
	int intersectionCount = 0;
	
	return intersectionCount;
}

__global__
void colorAtKernel(Tuple* colorBuffer, Ray* rays, int rayCount, Shape* shapes, int shapeCount) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int x = index; x < rayCount; x += stride) {
		int intersectionCount = intersectWorldCount(shapes, shapeCount, rays[x]);

		if (intersectionCount > 0) {
			colorBuffer[x] = { 1, 0, 0, 1 };
		}
		else {
			colorBuffer[x] = { 0, 0, 0, 1 };
		}
	}
}

void colorAt(Tuple* colorOut, World world, Ray* rays, int rayCount) {
	Tuple* colorBuffer;
	Shape* shapeBuffer;

	cudaMallocManaged(&colorBuffer, rayCount*sizeof(Tuple));
	cudaMallocManaged(&shapeBuffer, world.shapeCount*sizeof(Shape));
	memcpy(shapeBuffer, world.shapeArray, world.shapeCount*sizeof(Shape));

	int blockSize = 256;
	int numBlocks = (rayCount + blockSize - 1) / blockSize;
	colorAtKernel<<<numBlocks, blockSize>>>(colorBuffer, rays, rayCount, shapeBuffer, world.shapeCount);

	cudaDeviceSynchronize();
	cudaMemcpy(colorOut, colorBuffer, rayCount*sizeof(Tuple), cudaMemcpyDeviceToHost);

	cudaFree(colorBuffer);
	cudaFree(shapeBuffer);
}