#include <iostream>
#include <fstream>

__global__
void setCanvas(int canvasArea) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int n = index; n < canvasArea; n += stride) {

	}
}

const int SCREENWIDTH = 100;
const int SCREENHEIGHT = 50;

int main(int argn, const char** argv) {
	std::ofstream file;
	file.open(argv[1]);

	for (int y = 0; y < SCREENHEIGHT; y++) {
		for (int x = 0; x < SCREENWIDTH; x++) {

		}
	}

	return 0;
}