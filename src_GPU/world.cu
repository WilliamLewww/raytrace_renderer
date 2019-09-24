#include "world.h"
#include <stdio.h>

World createDefaultWorld() {
	World world;

	world.pointLightCount = 1;
	world.shapeCount = 2;

	world.lightArray = new PointLight[1];
	world.lightArray[0] = {{ -10, 10, -10, 1 }, { 1, 1, 1, 1 }};

	world.shapeArray = new Shape[2];
	world.shapeArray[0] = createSphere();
	world.shapeArray[0].modelMatrix = createIdentityMatrix(4);
	world.shapeArray[0].inverseModelMatrix = inverse(world.shapeArray[0].modelMatrix);

	world.shapeArray[1] = createSphere();
	world.shapeArray[1].modelMatrix = createIdentityMatrix(4);
	world.shapeArray[1].inverseModelMatrix = inverse(world.shapeArray[1].modelMatrix);

	return world;
}

__device__
int intersectWorldCount(Shape* shapes, int shapeCount, Ray ray, float* allInverseModelMatrixData) {
	int intersectionCount = 0;

	for (int x = 0; x < shapeCount; x++) {
		float* inverseModelMatrixData = new float[16];
		memcpy(inverseModelMatrixData, &allInverseModelMatrixData[x * 16], 16 * sizeof(float));

		intersectionCount += intersectCount(shapes[x], ray, inverseModelMatrixData);
	}

	return intersectionCount;
}

__global__
void colorAtKernel(Tuple* colorBuffer, Ray* rays, int rayCount, Shape* shapes, int shapeCount, float* allInverseModelMatrixData) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int x = index; x < rayCount; x += stride) {
		int intersectionCount = intersectWorldCount(shapes, shapeCount, rays[x], allInverseModelMatrixData);

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
	Ray* rayBuffer;
	Shape* shapeBuffer;
	float* allInverseModelMatrixData;

	cudaMallocManaged(&colorBuffer, rayCount*sizeof(Tuple));
	cudaMallocManaged(&shapeBuffer, world.shapeCount*sizeof(Shape));
	cudaMemcpy(shapeBuffer, world.shapeArray, world.shapeCount*sizeof(Shape), cudaMemcpyHostToDevice);

	cudaMallocManaged(&rayBuffer, rayCount*sizeof(Ray));
	cudaMemcpy(rayBuffer, rays, rayCount*sizeof(Ray), cudaMemcpyHostToDevice);

	cudaMallocManaged(&allInverseModelMatrixData, world.shapeCount*16*sizeof(float));

	for (int x = 0; x < world.shapeCount; x++) {
		cudaMemcpy(&allInverseModelMatrixData[(x * 16)], world.shapeArray[x].inverseModelMatrix.data, 16*sizeof(float), cudaMemcpyHostToDevice);
	}

	int blockSize = 256;
	int numBlocks = (rayCount + blockSize - 1) / blockSize;
	colorAtKernel<<<numBlocks, blockSize>>>(colorBuffer, rayBuffer, rayCount, shapeBuffer, world.shapeCount, allInverseModelMatrixData);

	cudaDeviceSynchronize();
	cudaMemcpy(colorOut, colorBuffer, rayCount*sizeof(Tuple), cudaMemcpyDeviceToHost);

	cudaFree(colorBuffer);
	cudaFree(shapeBuffer);
	cudaFree(allInverseModelMatrixData);

	cudaDeviceReset();
}