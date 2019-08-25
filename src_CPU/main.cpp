#include "canvas.h"
#include "tuple.h"
#include "matrix.h"

const int SCREENWIDTH = 100;
const int SCREENHEIGHT = 60;

int main(int argc, const char** argv) {
	Canvas* canvas = new Canvas(SCREENWIDTH, SCREENHEIGHT);
	canvas->saveToFile(argv[1]);

	Matrix4 matrixA = createMatrix4();
	Matrix4 matrixB = createMatrix4();
	matrixA[0][0] = 1;
	matrixA[0][1] = 2;
	matrixA[0][2] = 3;
	matrixA[0][3] = 4;
	matrixA[1][0] = 5;
	matrixA[1][1] = 6;
	matrixA[1][2] = 7;
	matrixA[1][3] = 8;
	matrixA[2][0] = 9;
	matrixA[2][1] = 8;
	matrixA[2][2] = 7;
	matrixA[2][3] = 6;
	matrixA[3][0] = 5;
	matrixA[3][1] = 4;
	matrixA[3][2] = 3;
	matrixA[3][3] = 2;

	matrixB[0][0] = -2;
	matrixB[0][1] = 1;
	matrixB[0][2] = 2;
	matrixB[0][3] = 3;
	matrixB[1][0] = 3;
	matrixB[1][1] = 2;
	matrixB[1][2] = 1;
	matrixB[1][3] = -1;
	matrixB[2][0] = 4;
	matrixB[2][1] = 3;
	matrixB[2][2] = 6;
	matrixB[2][3] = 5;
	matrixB[3][0] = 1;
	matrixB[3][1] = 2;
	matrixB[3][2] = 7;
	matrixB[3][3] = 8;

	std::cout << (matrixA * matrixB) << std::endl;
}