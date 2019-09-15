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

	world.shapeArray = new Shape[2];
	world.shapeArray[0] = createSphere();
	world.shapeArray[1] = createSphere();

	return world;
}

__device__
int intersectWorldCount(Shape* shapes, int shapeCount, Ray ray, float* allModelMatrixData) {
	int intersectionCount = 0;

	for (int x = 0; x < shapeCount; x++) {
		float* modelMatrixData = new float[16];
		memcpy(modelMatrixData, &allModelMatrixData[x * 16], 16 * sizeof(float));

		intersectionCount += intersectCount(shapes[x], ray, modelMatrixData);
	}
	
	return intersectionCount;
}

__global__
void colorAtKernel(Tuple* colorBuffer, Ray* rays, int rayCount, Shape* shapes, int shapeCount, float* allModelMatrixData) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int x = index; x < rayCount; x += stride) {
		int intersectionCount = intersectWorldCount(shapes, shapeCount, rays[x], allModelMatrixData);

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
	float* allModelMatrixData;

	cudaMallocManaged(&colorBuffer, rayCount*sizeof(Tuple));
	cudaMallocManaged(&shapeBuffer, world.shapeCount*sizeof(Shape));
	memcpy(shapeBuffer, world.shapeArray, world.shapeCount*sizeof(Shape));

	cudaMallocManaged(&allModelMatrixData, world.shapeCount*16*sizeof(float));

	for (int x = 0; x < world.shapeCount; x++) {
		memcpy(&allModelMatrixData[(x * 16)], world.shapeArray[x].modelMatrix.data, 16*sizeof(float));
	}

	int blockSize = 256;
	int numBlocks = (rayCount + blockSize - 1) / blockSize;
	colorAtKernel<<<numBlocks, blockSize>>>(colorBuffer, rays, rayCount, shapeBuffer, world.shapeCount, allModelMatrixData);

	cudaDeviceSynchronize();
	cudaMemcpy(colorOut, colorBuffer, rayCount*sizeof(Tuple), cudaMemcpyDeviceToHost);

	cudaFree(colorBuffer);
	cudaFree(shapeBuffer);
	cudaFree(allModelMatrixData);
}