#include "canvas.h"
#include "tuple.h"
#include "matrix.h"
#include "transform.h"
#include "ray.h"

const int SCREENWIDTH = 100;
const int SCREENHEIGHT = 60;

int main(int argc, const char** argv) {
	Canvas* canvas = new Canvas(SCREENWIDTH, SCREENHEIGHT);
	canvas->saveToFile(argv[1]);

	Ray r = createRay(createPoint(0, 0, -5), createVector(0, 0, 1));
	Sphere s = createSphere();

	int intersectionCount;
	Intersection* intersections = intersect(s, r, intersectionCount);
	
	std::cout << intersections[1] << std::endl;
}