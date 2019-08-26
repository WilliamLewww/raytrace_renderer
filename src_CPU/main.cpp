#include "canvas.h"
#include "tuple.h"
#include "matrix.h"
#include "transform.h"
#include "ray.h"
#include "sphere.h"

const int SCREENWIDTH = 250;
const int SCREENHEIGHT = 250;

int main(int argc, const char** argv) {
	// Canvas* canvas = new Canvas(SCREENWIDTH, SCREENHEIGHT);

	// Tuple cameraOrigin = createPoint(0, 0, -5);
	// Sphere sphere = createSphere();

	// for (int x = 0; x < SCREENWIDTH; x++) {
	// 	for (int y = 0; y < SCREENHEIGHT; y++) {
	// 		Tuple targetPosition = createPoint((float(x) / float(SCREENWIDTH) - 0.5) * 10, (float(y) / float(SCREENHEIGHT) - 0.5) * 10, 10.0);
	// 		Ray ray = createRay(cameraOrigin, normalize(targetPosition - cameraOrigin));
			
	// 		int intersectionCount;
	// 		Intersection* intersections = intersect(sphere, ray, intersectionCount);

	// 		if (intersectionCount > 0) { canvas->setPixel(x, y, {1.0, 0.0, 0.0, 1.0}); }
	// 		else { canvas->setPixel(x, y, {0.0, 0.0, 0.0, 1.0}); }
	// 	}
	// }

	// canvas->saveToFile(argv[1]);

	Sphere s = createSphere();
	s.modelMatrix = createScaleMatrix(1, 0.5, 1) * createRotationMatrixZ(M_PI / 5);
	Tuple n = normalAt(s, createPoint(0, sqrt(2) / 2, -sqrt(2) / 2));
	std::cout << n << std::endl;
}