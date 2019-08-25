#include "canvas.h"
#include "tuple.h"
#include "matrix.h"

const int SCREENWIDTH = 100;
const int SCREENHEIGHT = 60;

int main(int argc, const char** argv) {
	Canvas* canvas = new Canvas(SCREENWIDTH, SCREENHEIGHT);
	canvas->saveToFile(argv[1]);

	Matrix4 matrix = createMatrix4();
	std::cout << matrix << std::endl;
}