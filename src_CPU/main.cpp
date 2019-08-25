#include "canvas.h"
#include "tuple.h"
#include "matrix.h"

const int SCREENWIDTH = 100;
const int SCREENHEIGHT = 60;

int main(int argc, const char** argv) {
	Canvas* canvas = new Canvas(SCREENWIDTH, SCREENHEIGHT);
	canvas->saveToFile(argv[1]);

	Matrix matrixA = createMatrix(4, 4);
	matrixA[0][0] = -2;
	matrixA[0][1] = -8;
	matrixA[0][2] = 3;
	matrixA[0][3] = 5;
	matrixA[1][0] = -3;
	matrixA[1][1] = 1;
	matrixA[1][2] = 7;
	matrixA[1][3] = 3;
	matrixA[2][0] = 1;
	matrixA[2][1] = 2;
	matrixA[2][2] = -9;
	matrixA[2][3] = 6;
	matrixA[3][0] = -6;
	matrixA[3][1] = 7;
	matrixA[3][2] = 7;
	matrixA[3][3] = -9;

	std::cout << determinate(matrixA) << std::endl;
}