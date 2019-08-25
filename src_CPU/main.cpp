#include <fstream>
#include "canvas.h"

const int SCREENWIDTH = 100;
const int SCREENHEIGHT = 60;

int main(int argc, const char** argv) {
	std::ofstream file;
	file.open(argv[1]);

	file << "PPM\n" << SCREENWIDTH << " " << SCREENHEIGHT << "\n255\n";
	for (int y = SCREENHEIGHT - 1; y >= 0; y--) {
		for (int x = 0; x < SCREENWIDTH; x++) {
			int red = 255;
			int green = 255;
			int blue = 255;

			file << red << " " << green << " " << blue << "\n";
		}
	}

	file.close();
}