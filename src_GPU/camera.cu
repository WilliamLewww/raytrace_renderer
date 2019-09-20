#include "camera.h"

Camera createCamera(float viewWidth, float viewHeight, float fieldOfView) {
	Camera camera;

	camera.viewWidth = viewWidth;
	camera.viewHeight = viewHeight;
	camera.fieldOfView = fieldOfView;

	float halfView = tan(camera.fieldOfView / 2);
	float aspect = camera.viewWidth / camera.viewHeight;

	if (aspect >= 1) {
		camera.halfWidth = halfView;
		camera.halfHeight = halfView / aspect;
	}
	else {
		camera.halfWidth = halfView * aspect;
		camera.halfHeight = halfView;
	}

	camera.pixelSize = (camera.halfWidth * 2) / camera.viewWidth;

	camera.viewMatrix = createIdentityMatrix(4);

	return camera;
}

Matrix createViewMatrix(Tuple from, Tuple to, Tuple up) {
	Tuple forward = normalize(to - from);
	Tuple upN = normalize(up);
	Tuple left = cross(forward, upN);
	Tuple trueUp = cross(left, forward);

	Matrix orientation = createMatrix(4, 4);
	setDataMatrix(&orientation, 0, 0, left.x);
	setDataMatrix(&orientation, 0, 1, left.y);
	setDataMatrix(&orientation, 0, 2, left.z);

	setDataMatrix(&orientation, 1, 0, trueUp.x);
	setDataMatrix(&orientation, 1, 1, trueUp.y);
	setDataMatrix(&orientation, 1, 2, trueUp.z);

	setDataMatrix(&orientation, 2, 0, -forward.x);
	setDataMatrix(&orientation, 2, 1, -forward.y);
	setDataMatrix(&orientation, 2, 2, -forward.z);

	setDataMatrix(&orientation, 3, 3, 1);

	return orientation * createTranslateMatrix(-from.x, -from.y, -from.z);
}

__global__
void rayForPixelKernel(Ray* rayBuffer, float* inverseViewMatrixBuffer, Camera camera) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int x = index; x < camera.viewWidth * camera.viewHeight; x += stride) {
		int currentX = x % int(camera.viewWidth);
		int currentY = int(x / camera.viewWidth);

		float offsetX = (currentX + 0.5) * camera.pixelSize;
		float offsetY = (currentY + 0.5) * camera.pixelSize;

		float worldX = camera.halfWidth - offsetX;
		float worldY = camera.halfHeight - offsetY;

		Tuple worldPoint = { worldX, worldY, -1.0, 1.0 };
		Tuple zeroPoint = { 0.0, 0.0, 0.0, 1.0 };

		Matrix inverseViewMatrix = createMatrix(4, 4, inverseViewMatrixBuffer);

		Tuple pixel = inverseViewMatrix * worldPoint;
		Tuple origin = inverseViewMatrix * zeroPoint;
		Tuple direction = normalize(pixel - origin);

		rayBuffer[x] = { origin, direction };
	}
}

void rayForPixel(Ray* rayOut, Camera camera) {
	int count = camera.viewWidth * camera.viewHeight;
	Ray* rayBuffer;
	float* inverseViewMatrixBuffer;

	cudaMallocManaged(&rayBuffer, count*sizeof(Ray));
	cudaMallocManaged(&inverseViewMatrixBuffer, 16*sizeof(float));
	cudaMemcpy(inverseViewMatrixBuffer, camera.inverseViewMatrix.data, 16*sizeof(float), cudaMemcpyHostToDevice);

	int blockSize = 256;
	int numBlocks = (count + blockSize - 1) / blockSize;
	rayForPixelKernel<<<numBlocks, blockSize>>>(rayBuffer, inverseViewMatrixBuffer, camera);

	cudaDeviceSynchronize();

	cudaMemcpy(rayOut, rayBuffer, count*sizeof(Ray), cudaMemcpyDeviceToHost);

	cudaFree(rayBuffer);
	cudaFree(inverseViewMatrixBuffer);

	cudaDeviceReset();
}