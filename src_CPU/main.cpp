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

	Sphere s = createSphere();

	Intersection* intersections = new Intersection[4];
	intersections[0] = createIntersection(5, &s);
	intersections[1] = createIntersection(7, &s);
	intersections[2] = createIntersection(3, &s);
	intersections[3] = createIntersection(-2, &s);
	
	std::cout << *hit(intersections, 4) << std::endl;
}