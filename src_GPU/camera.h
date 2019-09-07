#pragma once
#include <cmath>
#include "ray.h"
#include "transform.h"
#include "analysis.h"

struct Camera {
	float hSize;
	float vSize;
	float fieldOfView;

	Matrix viewMatrix;
	Matrix inverseViewMatrix;

	float halfWidth;
	float halfHeight;
	float pixelSize;
};

Camera createCamera(float hSize, float vSize, float fieldOfView) {
	Camera camera = { hSize, vSize, fieldOfView, createIdentityMatrix(4) };

	float halfView = tan(camera.fieldOfView / 2);
	float aspect = camera.hSize / camera.vSize;

	if (aspect >= 1) {
		camera.halfWidth = halfView;
		camera.halfHeight = halfView / aspect;
	}
	else {
		camera.halfWidth = halfView * aspect;
		camera.halfHeight = halfView;
	}

	camera.pixelSize = (camera.halfWidth * 2) / camera.hSize;

	return camera;
}

Matrix createViewMatrix(Tuple from, Tuple to, Tuple up) {
	Tuple forward = normalize(to - from);
	Tuple upN = normalize(up);
	Tuple left = cross(forward, upN);
	Tuple trueUp = cross(left, forward);

	Matrix orientation = createMatrix(4, 4);
	orientation[0][0] = left.x; orientation[1][0] = trueUp.x;
	orientation[0][1] = left.y; orientation[1][1] = trueUp.y;
	orientation[0][2] = left.z; orientation[1][2] = trueUp.z;
	orientation[0][3] = 0; 		orientation[1][3] = 0;

	orientation[2][0] = -forward.x; orientation[3][0] = 0;
	orientation[2][1] = -forward.y; orientation[3][1] = 0;
	orientation[2][2] = -forward.z; orientation[3][2] = 0;
	orientation[2][3] = 0; 			orientation[3][3] = 1;

	return orientation * createTranslateMatrix(-from.x, -from.y, -from.z);
}

Ray rayForPixel(Camera camera, int x, int y) {
	float offsetX = (x + 0.5) * camera.pixelSize;
	float offsetY = (y + 0.5) * camera.pixelSize;

	float worldX = camera.halfWidth - offsetX;
	float worldY = camera.halfHeight - offsetY;

	Tuple pixel = camera.inverseViewMatrix * createPoint(worldX, worldY, -1);
	Tuple origin = camera.inverseViewMatrix * createPoint(0, 0, 0);

	Tuple direction = normalize(pixel - origin);

	return createRay(origin, direction);
}

Matrix computeInverseViewMatrix(Camera camera) {
	return inverse(camera.viewMatrix);
}

__global__
void rayForPixelKernel(Ray* rayBuffer, int count, Camera camera) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int x = index; x < count; x += stride) {
		rayBuffer[x].origin.x = x;
	}
}

void rayForPixelGPU(Camera camera, int width, int height) {
	int count = width * height;
	Ray* rayBuffer;

	cudaMallocManaged(&rayBuffer, count*sizeof(Ray));

	int blockSize = 256;
	int numBlocks = (count + blockSize - 1) / blockSize;
	rayForPixelKernel<<<numBlocks, blockSize>>>(rayBuffer, count, camera);

	cudaDeviceSynchronize();

	cudaFree(rayBuffer);
}