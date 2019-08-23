#include <fstream>

int main(int argc, const char** argv) {
	std::ofstream file;

	file.open(argv[1]);

	int screenWidth = 400;
	int screenHeight = 250;

	file << "PPM\n" << screenWidth << " " << screenHeight << "\n255\n";

	for (int y = 0; y < screenHeight; y++) {
		for (int x = 0; x < screenWidth; x++) {
			int red = 255;
			int green = 255;
			int blue = 255;

			file << red << " " << green << " " << blue << "\n";
		}
	}
}