#include "canvas.h"
#include "tuple.h"
#include "matrix.h"
#include "transform.h"
#include "ray.h"
#include "world.h"
#include "camera.h"

const int SCREENWIDTH = 256;
const int SCREENHEIGHT = 256;

Canvas* render(Camera camera, World world) {
	Canvas* canvas = new Canvas(camera.hSize, camera.vSize);

	for (int y = 0; y < camera.vSize; y++) {
		for (int x = 0; x < camera.hSize; x++) {
			Ray ray = rayForPixel(camera, x, y);
			Tuple color = colorAt(world, ray);

			canvas->setPixel(x, y, color);
		}
	}

	return canvas;
}

int main(int argc, const char** argv) {
	// std::cout << "generating ray traced image (" << SCREENWIDTH << "x" << SCREENHEIGHT << ")" << std::endl;

	// Canvas* canvas = new Canvas(SCREENWIDTH, SCREENHEIGHT);

	// Tuple cameraOrigin = createPoint(0, 0, -5);
	// Sphere sphere = createSphere();
	// sphere.material = createMaterial();
	// sphere.material.color = createColor(1.0, 0.2, 1.0);

	// PointLight light = createPointLight(createPoint(-10, 10, -10), createColor(1, 1, 1));

	// for (int x = 0; x < SCREENWIDTH; x++) {
	// 	for (int y = 0; y < SCREENHEIGHT; y++) {
	// 		Tuple targetPosition = createPoint((float(x) / float(SCREENWIDTH) - 0.5) * 10, (float(y) / float(SCREENHEIGHT) - 0.5) * 10, 10.0);
	// 		Ray ray = createRay(cameraOrigin, normalize(targetPosition - cameraOrigin));
			
	// 		int intersectionCount;
	// 		Intersection* intersections = intersect(sphere, ray, intersectionCount);

	// 		if (intersectionCount > 0) {
	// 			Intersection* closestHit = hit(intersections, intersectionCount);
	// 			Precomputed computations = prepareComputations(*closestHit, ray);

	// 			Tuple color = lighting(closestHit->object->material, light, computations.point, computations.eyeV, computations.normalV);

	// 			canvas->setPixel(x, y, color); 
	// 		}
	// 		else { canvas->setPixel(x, y, {0.0, 0.0, 0.0, 1.0}); }
	// 	}
	// }

	// canvas->saveToFile(argv[1]);

	// std::cout << "finished generation" << std::endl;

	World world = createDefaultWorld();
	Camera camera = createCamera(11, 11, M_PI / 2);
	Tuple from = createPoint(0, 0, -5);
	Tuple to = createPoint(0, 0, 0);
	Tuple up = createVector(0, 1, 0);
	camera.viewMatrix = createViewMatrix(from, to, up);

	Canvas* canvas = render(camera, world);
	std::cout << canvas->getPixel(5, 5) << std::endl;
}