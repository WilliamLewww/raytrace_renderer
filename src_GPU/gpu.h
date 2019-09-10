#pragma once
#include <thrust/sort.h>

__device__
float dotGPU(Tuple tupleA, Tuple tupleB) {
	return ((tupleA.x * tupleB.x) + (tupleA.y * tupleB.y) + (tupleA.z * tupleB.z) + (tupleA.w * tupleB.w));
}

__device__
Tuple multiplyMatrixGPU(Matrix matrix, Tuple tuple) {
	return {
		((matrix.data[0][0] * tuple.x) + (matrix.data[0][1] * tuple.y) + (matrix.data[0][2] * tuple.z) + (matrix.data[0][3] * tuple.w)),
 		((matrix.data[1][0] * tuple.x) + (matrix.data[1][1] * tuple.y) + (matrix.data[1][2] * tuple.z) + (matrix.data[1][3] * tuple.w)),
 		((matrix.data[2][0] * tuple.x) + (matrix.data[2][1] * tuple.y) + (matrix.data[2][2] * tuple.z) + (matrix.data[2][3] * tuple.w)),
 		((matrix.data[3][0] * tuple.x) + (matrix.data[3][1] * tuple.y) + (matrix.data[3][2] * tuple.z) + (matrix.data[3][3] * tuple.w))
	};
}

__device__
Ray transformGPU(Ray ray, Matrix matrix) {
	return { multiplyMatrixGPU(matrix, ray.origin), multiplyMatrixGPU(matrix, ray.direction) };
}

__device__
Matrix createMatrixGPU(int rowCount, int columnCount) {
	Matrix temp;
	temp.data = new float*[rowCount];
	for (int x = 0; x < rowCount; x++) {
		temp.data[x] = new float[columnCount];
		for (int y = 0; y < columnCount; y++) {
			temp.data[x][y] = 0.0;
		}
	}

	temp.rowCount = rowCount;
	temp.columnCount = columnCount;

	return temp;
}

__device__
float cofactorGPU(Matrix matrix, int row, int column);

__device__
float determinateGPU(Matrix matrix) {
	if (matrix.rowCount == 2 && matrix.columnCount == 2) {
		return ((matrix.data[0][0] * matrix.data[1][1]) - (matrix.data[1][0] * matrix.data[0][1]));
	}
	
	float det = 0;
	for (int y = 0; y < matrix.columnCount; y++) {
		det += matrix.data[0][y] * cofactorGPU(matrix, 0, y);
	}

	return det;
}

__device__
Matrix submatrixGPU(Matrix matrix, int row, int column) {
	Matrix temp = createMatrixGPU(matrix.rowCount - 1, matrix.columnCount - 1);

	int currentX = 0;
	int currentY = 0;
	for (int x = 0; x < matrix.rowCount; x++) {
		for (int y = 0; y < matrix.columnCount; y++) {
			if (x != row && y != column) {
				temp.data[currentX][currentY] = matrix.data[x][y];
				if (currentY + 1 == temp.columnCount) {
					currentX += 1;
					currentY = 0;
				}
				else {
					currentY += 1;
				}
			}
		}
	}

	return temp;
}

__device__
float matrixMinorGPU(Matrix matrix, int row, int column) {
	Matrix temp = submatrixGPU(matrix, row, column);
	return determinateGPU(temp);
}

__device__
float cofactorGPU(Matrix matrix, int row, int column) {
	if ((row + column) % 2 == 0) {
		return matrixMinorGPU(matrix, row, column);
	}

	return -matrixMinorGPU(matrix, row, column);
}

__device__
Matrix inverseGPU(Matrix matrix) {
	Matrix temp = createMatrixGPU(matrix.rowCount, matrix.columnCount);
	float det = determinateGPU(matrix);

	for (int x = 0; x < temp.rowCount; x++) {
		for (int y = 0; y < temp.columnCount; y++) {
			float c = cofactorGPU(matrix, x, y);

			temp.data[y][x] = c / det;
		}
	}

	return temp;
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

__device__
Matrix createMatrixGPU(int rowCount, int columnCount, float* matrixBuffer, int offset = 0) {
	Matrix temp;
	temp.data = new float*[rowCount];
	for (int x = 0; x < rowCount; x++) {
		temp.data[x] = new float[columnCount];
		for (int y = 0; y < columnCount; y++) {
			temp.data[x][y] = matrixBuffer[(x * 4) + y + offset];
		}
	}

	temp.rowCount = rowCount;
	temp.columnCount = columnCount;

	return temp;
}

__global__
void rayForPixelKernel(Ray* rayBuffer, float* inverseViewMatrixBuffer, int count, Camera camera) {
	Matrix inverseViewMatrix = createMatrixGPU(4, 4, inverseViewMatrixBuffer);

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int x = index; x < count; x += stride) {
		int currentX = x % int(camera.hSize);
		int currentY = int(x / camera.hSize);

		float offsetX = (currentX + 0.5) * camera.pixelSize;
		float offsetY = (currentY + 0.5) * camera.pixelSize;

		float worldX = camera.halfWidth - offsetX;
		float worldY = camera.halfHeight - offsetY;

		Tuple pixel = multiplyMatrixGPU(inverseViewMatrix, { worldX, worldY, -1, 1 });
		Tuple origin = multiplyMatrixGPU(inverseViewMatrix, { 0, 0, 0, 1 });

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

__device__
Intersection* intersectSphereGPU(Shape shape, Ray ray, int& intersectionCount) {
	Tuple sphereToRay = subtractTuple(ray.origin, shape.origin);
	float a = dotGPU(ray.direction, ray.direction);
	float b = 2 * dotGPU(ray.direction, sphereToRay);
	float c = dotGPU(sphereToRay, sphereToRay) - 1;

	float discriminant = pow(b, 2) - (4 * a * c);

	if (discriminant < 0) {
		intersectionCount = 0;

		return nullptr;
	}

	if ((-b - sqrt(discriminant)) / (2 * a) == (-b + sqrt(discriminant)) / (2 * a)) {
		intersectionCount = 1;

		Intersection* intersection = new Intersection[1];
		intersection[0] = { (-b - sqrt(discriminant)) / (2 * a), &shape };

		return intersection;
	}

	intersectionCount = 2;

	Intersection* intersection = new Intersection[2];
	intersection[0] = { (-b - sqrt(discriminant)) / (2 * a), &shape };
	intersection[1] = { (-b + sqrt(discriminant)) / (2 * a), &shape };
	return intersection;
}

__device__
int intersectCountGPU(Shape shape, Ray ray, Matrix modelMatrix) {
	// Intersection* intersections;

	// Ray rayTransformed = transformGPU(ray, inverseGPU(modelMatrix));
	// if (shape.shape == SHAPES_SPHERE) {
	// 	intersections = intersectSphereGPU(shape, rayTransformed, intersectionCount);
	// }
	// if (shape.shape == SHAPES_PLANE) {
	// 	intersections = intersectPlane(shape, rayTransformed, intersectionCount);
	// }

	// return intersections;

	return 0;
}

__device__
int intersectWorldCountGPU(World world, Ray ray, Shape* shapeArrayBuffer, float* modelMatrixBuffer) {
	int totalIntersectionCount = 0;
	for (int x = 0; x < world.shapeCount; x++) {
		Matrix modelMatrix = createMatrixGPU(4, 4, modelMatrixBuffer, x * 16);
		totalIntersectionCount += intersectCountGPU(shapeArrayBuffer[x], ray, modelMatrix);;
	}

	return totalIntersectionCount;
}

__device__
bool sortIntersectionsGPU(Intersection intersectionA, Intersection intersectionB) {
    return intersectionA.t >= intersectionB.t;
}

__global__
void colorAtKernel(Tuple* colorBuffer, int count, World world, Ray* rays, Shape* shapeArrayBuffer, float* modelMatrixBuffer) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int x = index; x < count; x += stride) {
		int intersectionCount = intersectWorldCountGPU(world, rays[x], shapeArrayBuffer, modelMatrixBuffer);
		if (intersectionCount > 0) {
			colorBuffer[x] = { 1, 0, 0, 1 };
		}
		else {
			colorBuffer[x] = { 0, 0, 0, 1 };
		}
	}
}

void colorAtGPU(Tuple* colorOut, World world, Ray* rays, int count) {
	Tuple* colorBuffer;
	Shape* shapeArrayBuffer;
	float* modelMatrixBuffer;

	cudaMallocManaged(&colorBuffer, count*sizeof(Tuple));
	cudaMallocManaged(&shapeArrayBuffer, sizeof(Shape) * world.shapeCount);
	cudaMallocManaged(&modelMatrixBuffer, sizeof(float) * 16 * world.shapeCount);

	for (int x = 0; x < world.shapeCount; x++) {
		shapeArrayBuffer[x] = world.shapeArray[x];
		for (int y = 0; y < 4; y++) {
			for (int z = 0; z < 4; z++) {
				modelMatrixBuffer[(x * 16) + (y * 4) + z] = world.shapeArray[x].modelMatrix[y][z];
			}
		}
	}

	int blockSize = 256;
	int numBlocks = (count + blockSize - 1) / blockSize;
	colorAtKernel<<<numBlocks, blockSize>>>(colorBuffer, count, world, rays, shapeArrayBuffer, modelMatrixBuffer);

	cudaDeviceSynchronize();

	cudaMemcpy(colorOut, colorBuffer, count*sizeof(Tuple), cudaMemcpyDeviceToHost);

	cudaFree(colorBuffer);
	cudaFree(shapeArrayBuffer);
	cudaFree(modelMatrixBuffer);
}