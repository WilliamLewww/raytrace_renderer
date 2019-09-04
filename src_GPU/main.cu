#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include "canvas.h"
#include "world.h"
#include "camera.h"
#include "analysis.h"

const int SCREENWIDTH = 500;
const int SCREENHEIGHT = 250;

Canvas* render(Camera camera, World world) {
	Canvas* canvas = new Canvas(camera.hSize, camera.vSize);

	std::cout << "rendering ray traced image..." << std::endl;

	for (int y = 0; y < camera.vSize; y++) {
		for (int x = 0; x < camera.hSize; x++) {
			Ray ray = rayForPixel(camera, x, y);
			Tuple color = colorAt(world, ray);

			canvas->setPixel(x, y, color);
		}
	}

	std::cout << "finished rendering" << std::endl;

	return canvas;
}

int main(int argc, const char** argv) {
	Analysis::setAbsoluteStart();
	Analysis::createLabel(0, "rayForPixel");

	World world = createWorld2();
	
	Camera camera = createCamera(SCREENWIDTH, SCREENHEIGHT, M_PI / 3);
	Tuple from = createPoint(3, 2.5, -5);
	Tuple to = createPoint(0, 1, 0);
	Tuple up = createVector(0, 1, 0);
	camera.viewMatrix = createViewMatrix(from, to, up);
	camera.inverseViewMatrix = computeInverseViewMatrix(camera);

	Canvas* canvas = render(camera, world);
	canvas->saveToFile(argv[1]);
	std::cout << "saved image as: [" << argv[1] << "]" << std::endl;

	Analysis::printAll();
	Analysis::saveToFile(argv[2], SCREENWIDTH, SCREENHEIGHT);

	delete canvas;
	return 0;
}