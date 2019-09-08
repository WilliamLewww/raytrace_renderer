#pragma once

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

void rayForPixelGPU(Ray* rayOut, Camera camera, int width, int height) {
	int count = width * height;
	Ray* rayBuffer;
	float* inverseViewMatrixBuffer;

	cudaMallocManaged(&rayBuffer, count*sizeof(Ray));
	cudaMallocManaged(&inverseViewMatrixBuffer, sizeof(float)*16);

	for (int x = 0; x < 4; x++) {
		for (int y = 0; y < 4; y++) {
			inverseViewMatrixBuffer[(x * 4) + y] = camera.inverseViewMatrix[x][y];
		}
	}

	int blockSize = 256;
	int numBlocks = (count + blockSize - 1) / blockSize;
	rayForPixelKernel<<<numBlocks, blockSize>>>(rayBuffer, inverseViewMatrixBuffer, count, camera);

	cudaDeviceSynchronize();

	cudaMemcpy(rayOut, rayBuffer, count*sizeof(Ray), cudaMemcpyDeviceToHost);

	cudaFree(rayBuffer);
	cudaFree(inverseViewMatrixBuffer);
}

void colorAtGPU(Tuple* colorOut, World world, Ray* rays, int width, int height) {
	int count = width * height;
	Tuple* tupleBuffer;

	cudaMallocManaged(&tupleBuffer, count*sizeof(Tuple));

	cudaFree(tupleBuffer);
}