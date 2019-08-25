#pragma once

class Canvas{
private:
	int width;
	int height;

	float** data;
public:
	inline Canvas(int width, int height) {
		this->width = width;
		this->height = height;

		data = new float*[width];
		for (int x = 0; x < width; x++) {
			data[width] = new float[height];
		}
	}

	inline ~Canvas() {
		for (int x = 0; x < width; x++) {
			delete [] data[width];
		}

		delete [] data;
	}
};