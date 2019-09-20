#pragma once
#include <fstream>
#include "tuple.h"

struct Canvas {
	Tuple* data;

	int width;
	int height;
};

Canvas createCanvas(const int width, const int height);
void setPixelCanvas(Canvas* canvas, int x, int y, Tuple color);
Tuple getPixelCanvas(Canvas* canvas, int x, int y);
void saveCanvasToFile(Canvas* canvas, const char* filename);