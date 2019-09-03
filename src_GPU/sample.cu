#include <iostream>

__global__
void add(int n, float *x, float *y) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride) {
		y[i] = x[i] + y[i];
	}
}

int main() {
	int count = 100;
	float *first, *second;

	cudaMallocManaged(&first, count*sizeof(float));
	cudaMallocManaged(&second, count*sizeof(float));

	for (int x = 0; x < count; x++) {
		first[x] = x;
		second[x] = x;
	}

	int blockSize = 256;
	int numBlocks = (count + blockSize - 1) / blockSize;
	add<<<numBlocks, blockSize>>>(count, first, second);

	cudaDeviceSynchronize();

	for (int x = 0; x < count; x++) {
		std::cout << second[x] << std::endl;
	}

	cudaFree(first);
	cudaFree(second);

	return 0;
}