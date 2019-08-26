#pragma once
#include <fstream>
#include "tuple.h"

class Canvas{
private:
	int width;
	int height;

	Tuple** pixelData;
public:
	inline Canvas(int width, int height) {
		this->width = width;
		this->height = height;

		pixelData = new Tuple*[width];
		for (int x = 0; x < width; x++) {
			pixelData[x] = new Tuple[height];
			for (int y = 0; y < height; y++) {
				pixelData[x][y] = createColor(0.0, 0.0, 0.0);
			}
		}
	}

	inline ~Canvas() {
		for (int x = 0; x < width; x++) {
			delete [] pixelData[width];
		}

		delete [] pixelData;
	}

	inline void setPixel(int x, int y, Tuple color) {
		pixelData[x][y] = color;
	}

	Tuple getPixel(int x, int y) {
		return pixelData[x][y];
	}

	void saveToFile(const char* filename) {
		std::ofstream file;
		file.open(filename);

		file << "P3\n" << width << " " << height << "\n255\n";
		for (int y = height - 1; y >= 0; y--) {
			for (int x = 0; x < width; x++) {
				Tuple color = getPixel(x, y) * 255;

				if (color.x > 255) { color.x = 255; }
				if (color.y > 255) { color.y = 255; }
				if (color.z > 255) { color.z = 255; }

				file << (int)color.x << " " << (int)color.y << " " << (int)color.z << "\n";
			}
		}

		file.close();
	}
};