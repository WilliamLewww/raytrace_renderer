#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

#include "analysis.h"
#include "canvas.h"
#include "camera.h"
#include "world.h"

const int SCREENWIDTH = 100;
const int SCREENHEIGHT = 50;

Canvas render(Camera camera, World world) {
	Canvas canvas = createCanvas(camera.viewWidth, camera.viewHeight);

	std::cout << "rendering ray traced image..." << std::endl;

	Ray* rayOut = new Ray[int(camera.viewWidth * camera.viewHeight)];
	rayForPixel(rayOut, camera);

	Tuple* colorOut = new Tuple[int(camera.viewWidth * camera.viewHeight)];
	colorAt(colorOut, world, rayOut, int(camera.viewWidth * camera.viewHeight));

	for (int y = 0; y < camera.viewHeight; y++) {
		for (int x = 0; x < camera.viewWidth; x++) {
			setPixelCanvas(&canvas, x, y, colorOut[int((y * camera.viewWidth) + x)]);
		}
	}

	std::cout << "finished rendering" << std::endl;

	return canvas;
}

int main(int argc, const char** argv) {
	Analysis::setAbsoluteStart();

	World world = createDefaultWorld();
	Camera camera = createCamera(SCREENWIDTH, SCREENHEIGHT, M_PI / 3);

	Tuple from = { 3.0, 2.5, -5.0, 1.0 };
	Tuple to = { 0.0, 1.0, 0.0, 1.0 };
	Tuple up = { 0.0, 2.0, 0.0, 0.0 };
	camera.viewMatrix = createViewMatrix(from, to, up);
	camera.inverseViewMatrix = inverse(camera.viewMatrix);

	Canvas canvas = render(camera, world);
	saveCanvasToFile(&canvas, argv[1]);
	std::cout << "saved image as : [" << argv[1] << "]" << std::endl;

	Analysis::printAll();
	Analysis::saveToFile(argv[2], SCREENWIDTH, SCREENHEIGHT);

	cudaDeviceReset();

	return 0;
}