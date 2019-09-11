#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include "analysis.h"
#include "canvas.h"

const int SCREENWIDTH = 100;
const int SCREENHEIGHT = 50;

int main(int argc, const char** argv) {
	Analysis::setAbsoluteStart();

	Canvas canvas = createCanvas(SCREENWIDTH, SCREENHEIGHT);
	saveCanvasToFile(&canvas, argv[1]);

	Analysis::printAll();
	Analysis::saveToFile(argv[2], SCREENWIDTH, SCREENHEIGHT);
	return 0;
}