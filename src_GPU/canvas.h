#pragma once
#include <fstream>
#include "tuple.h"

struct Canvas {
	Tuple* pixelData;

	int width;
	int height;
};

Canvas createCanvas(const int width, const int height) {
	Canvas canvas;

	canvas.width = width;
	canvas.height = height;

	canvas.pixelData = new Tuple[canvas.width * canvas.height];

	for (int y = 0; y < canvas.height; y++) {
		for (int x = 0; x < canvas.width; x++) {
			canvas.pixelData[(y * canvas.width) + x] = { 0, 0, 0, 1 };
		}
	}

	return canvas;
}

void setPixelCanvas(Canvas* canvas, int x, int y, Tuple color) {
	canvas->pixelData[(y * canvas->width) + x] = color;
}

Tuple getPixelCanvas(Canvas* canvas, int x, int y) {
	return canvas->pixelData[(y * canvas->width) + x];
}

void saveCanvasToFile(Canvas* canvas, const char* filename) {
	std::ofstream file;
	file.open(filename);

	file << "P3\n" << canvas->width << " " << canvas->height << "\n255\n";

	for (int y = 0; y < canvas->height; y++) {
		for (int x = 0; x < canvas->width; x++) {
			Tuple pixel = getPixelCanvas(canvas, x, y) * 255;

			if (pixel.x > 255) { pixel.x = 255; }
			if (pixel.y > 255) { pixel.y = 255; }
			if (pixel.z > 255) { pixel.z = 255; }

			file << (int)pixel.x << " " << (int)pixel.y << " " << (int)pixel.z << "\n";
		}
	}

	file.close();
}