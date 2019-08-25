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

		file << "PPM\n" << width << " " << height << "\n255\n";
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				Tuple color = getPixel(x, y) * 255.0;

				file << color.x << " " << color.y << " " << color.z << "\n";
			}
		}

		file.close();
	}
};