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

__device__
Tuple multiplyMatrixTuple(float* matrix, Tuple tuple) {
	return {
		((matrix[0] * tuple.x) + (matrix[1] * tuple.y) + (matrix[2] * tuple.z) + (matrix[3] * tuple.w)),
 		((matrix[4] * tuple.x) + (matrix[5] * tuple.y) + (matrix[6] * tuple.z) + (matrix[7] * tuple.w)),
 		((matrix[8] * tuple.x) + (matrix[9] * tuple.y) + (matrix[10] * tuple.z) + (matrix[11] * tuple.w)),
 		((matrix[12] * tuple.x) + (matrix[13] * tuple.y) + (matrix[14] * tuple.z) + (matrix[15] * tuple.w))
	};
}

__device__
float magnitudeTuple(Tuple tuple) {
	return sqrt(pow(tuple.x, 2) + pow(tuple.y, 2) + pow(tuple.z, 2) + pow(tuple.w, 2));
}

__device__
Tuple normalizeTuple(Tuple tuple) {
	return { tuple.x / magnitudeTuple(tuple), tuple.y / magnitudeTuple(tuple), tuple.z / magnitudeTuple(tuple), tuple.w / magnitudeTuple(tuple) };
}

__device__
Tuple subtractTuple(Tuple tupleA, Tuple tupleB) {
	return { tupleA.x - tupleB.x, tupleA.y - tupleB.y, tupleA.z - tupleB.z, tupleA.w - tupleB.w };
}

__global__
void rayForPixelKernel(Ray* rayBuffer, float* inverseViewMatrixBuffer, int count, Camera camera) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int x = index; x < count; x += stride) {
		int currentX = x % int(camera.hSize);
		int currentY = int(x / camera.hSize);

		float offsetX = (currentX + 0.5) * camera.pixelSize;
		float offsetY = (currentY + 0.5) * camera.pixelSize;

		float worldX = camera.halfWidth - offsetX;
		float worldY = camera.halfHeight - offsetY;

		Tuple pixel = multiplyMatrixTuple(inverseViewMatrixBuffer, { worldX, worldY, -1, 1 });
		Tuple origin = multiplyMatrixTuple(inverseViewMatrixBuffer, { 0, 0, 0, 1 });

		Tuple direction = normalizeTuple(subtractTuple(pixel, origin));

		rayBuffer[x] = { origin, direction };
	}
}

void rayForPixelGPU(Camera camera, int width, int height) {
	int count = width * height;
	Ray* rayBuffer;
	float* inverseViewMatrixBuffer;

	Analysis::begin();
	cudaMallocManaged(&rayBuffer, count*sizeof(Ray));
	cudaMallocManaged(&inverseViewMatrixBuffer, sizeof(float)*16);
	Analysis::end(1);

	for (int x = 0; x < 4; x++) {
		for (int y = 0; y < 4; y++) {
			inverseViewMatrixBuffer[(x * 4) + y] = camera.inverseViewMatrix[x][y];
		}
	}

	Analysis::begin();
	int blockSize = 256;
	int numBlocks = (count + blockSize - 1) / blockSize;
	rayForPixelKernel<<<numBlocks, blockSize>>>(rayBuffer, inverseViewMatrixBuffer, count, camera);

	cudaDeviceSynchronize();

	cudaFree(rayBuffer);
	cudaFree(inverseViewMatrixBuffer);
	Analysis::end(2);
}