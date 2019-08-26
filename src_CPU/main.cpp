#include "canvas.h"
#include "tuple.h"
#include "matrix.h"
#include "transform.h"
#include "ray.h"

const int SCREENWIDTH = 250;
const int SCREENHEIGHT = 250;

int main(int argc, const char** argv) {
	Canvas* canvas = new Canvas(SCREENWIDTH, SCREENHEIGHT);

	Tuple cameraOrigin = createPoint(0, 0, -5);
	Sphere sphere = createSphere();

	for (int x = 0; x < SCREENWIDTH; x++) {
		for (int y = 0; y < SCREENHEIGHT; y++) {
			Ray ray = createRay(cameraOrigin, createPoint((float(x) / float(SCREENWIDTH) - 0.5) * 10, (float(y) / float(SCREENHEIGHT) - 0.5) * 10, 10.0) - cameraOrigin);
			
			int intersectionCount;
			Intersection* intersections = intersect(sphere, ray, intersectionCount);

			if (intersectionCount > 0) {
				canvas->setPixel(x, y, {1.0, 0.0, 0.0, 1.0});
			}
			else {
				canvas->setPixel(x, y, {0.0, 0.0, 0.0, 1.0});
			}
		}
	}
	canvas->saveToFile(argv[1]);
}