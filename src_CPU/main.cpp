#include "canvas.h"
#include "tuple.h"
#include "matrix.h"
#include "transform.h"
#include "ray.h"
#include "world.h"

const int SCREENWIDTH = 256;
const int SCREENHEIGHT = 256;

int main(int argc, const char** argv) {
	std::cout << "Generating Ray Traced Image! (" << SCREENWIDTH << "x" << SCREENHEIGHT << ")" << std::endl;

	Canvas* canvas = new Canvas(SCREENWIDTH, SCREENHEIGHT);

	Tuple cameraOrigin = createPoint(0, 0, -5);
	Sphere sphere = createSphere();
	sphere.material = createMaterial();
	sphere.material.color = createColor(1.0, 0.2, 1.0);

	PointLight light = createPointLight(createPoint(-10, 10, -10), createColor(1, 1, 1));

	for (int x = 0; x < SCREENWIDTH; x++) {
		for (int y = 0; y < SCREENHEIGHT; y++) {
			Tuple targetPosition = createPoint((float(x) / float(SCREENWIDTH) - 0.5) * 10, (float(y) / float(SCREENHEIGHT) - 0.5) * 10, 10.0);
			Ray ray = createRay(cameraOrigin, normalize(targetPosition - cameraOrigin));
			
			int intersectionCount;
			Intersection* intersections = intersect(sphere, ray, intersectionCount);

			if (intersectionCount > 0) {
				Intersection* closestHit = hit(intersections, intersectionCount);
				Tuple point = project(ray, closestHit->t);
				Tuple normal = normalAt(*closestHit->object, point);
				Tuple eye = negate(ray.direction);

				Tuple color = lighting(closestHit->object->material, light, point, eye, normal);

				canvas->setPixel(x, y, color); 
			}
			else { canvas->setPixel(x, y, {0.0, 0.0, 0.0, 1.0}); }
		}
	}

	canvas->saveToFile(argv[1]);

	std::cout << "Finished Generation!" << std::endl;

	World world = createDefaultWorld();
	Ray ray = createRay(createPoint(0, 0, -5), createVector(0, 0, 1));

	int intersectionCount;
	Intersection* intersections = intersectWorld(world, ray, intersectionCount);
	std::cout << intersectionCount << std::endl;
}