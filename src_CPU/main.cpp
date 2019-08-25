#include "canvas.h"
#include "tuple.h"
#include "matrix.h"

const int SCREENWIDTH = 100;
const int SCREENHEIGHT = 60;

int main(int argc, const char** argv) {
	Canvas* canvas = new Canvas(SCREENWIDTH, SCREENHEIGHT);
	canvas->saveToFile(argv[1]);

	Matrix matrixA = createMatrix(3, 3);
	matrixA[0][0] = 3;
	matrixA[0][1] = 5;
	matrixA[0][2] = 0;
	matrixA[1][0] = 2;
	matrixA[1][1] = -1;
	matrixA[1][2] = -7;
	matrixA[2][0] = 6;
	matrixA[2][1] = -1;
	matrixA[2][2] = 5;


	std::cout << cofactor(matrixA, 0, 0) << std::endl;
}