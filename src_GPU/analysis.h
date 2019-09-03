#pragma once
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <ctime>
#include <utility>

class Analysis {
private:
	static std::chrono::high_resolution_clock::time_point start;
	static std::chrono::high_resolution_clock::time_point finish;

	static std::vector<std::vector<int64_t>> durationList;
	static std::vector<std::pair<int, const char*>> labelList;
public:
	inline static void begin() { start = std::chrono::high_resolution_clock::now(); }
	inline static void end(int index) { 
		finish = std::chrono::high_resolution_clock::now(); 

		if (index >= durationList.size()) {
			durationList.push_back(std::vector<int64_t>());
		}

		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
		durationList[index].push_back(duration);
	}

	inline static void createLabel(int index, const char* label) {
		std::pair<int, const char*> tempLabel;
		tempLabel.first = index;
		tempLabel.second = label;

		labelList.push_back(tempLabel);
	}

	inline static void printAll() {
		std::cout << std::endl;

		int64_t averageT = 0;
		for (int x = 0; x < durationList.size(); x++) {
			int64_t average = 0;

			for (int y = 0; y < durationList[x].size(); y++) {
				average += durationList[x][y];
			}

			for (int z = 0; z < labelList.size(); z++) {
				if (labelList[z].first == x) {
					std::cout << labelList[z].second << " ";
				}
			}

			std::cout << "[" << x << "]: ";
			std::cout << average / durationList[x].size() << std::endl;

			averageT += average / durationList[x].size();
		}

		std::cout << "Total: " << averageT << std::endl;
		std::cout << std::endl;
	}

	inline static void saveToFile(const char* filename, const int screenWidth, const int screenHeight) {
		std::ofstream file;
		file.open(filename, std::ios::app);

		time_t tempTime = time(NULL);
		file << ctime(&tempTime);
		file << "image resolution: " << screenWidth << "x" << screenHeight << std::endl;

		int64_t averageT = 0;
		for (int x = 0; x < durationList.size(); x++) {
			int64_t average = 0;

			for (int y = 0; y < durationList[x].size(); y++) {
				average += durationList[x][y];
			}

			for (int z = 0; z < labelList.size(); z++) {
				if (labelList[z].first == x) {
					file << labelList[z].second << " ";
				}
			}

			file << "[" << x << "]: ";
			file << average / durationList[x].size() << std::endl;

			averageT += average / durationList[x].size();
		}

		file << "Total: " << averageT << std::endl;
		file << std::endl;
	}
};

std::chrono::high_resolution_clock::time_point Analysis::start;
std::chrono::high_resolution_clock::time_point Analysis::finish;

std::vector<std::vector<int64_t>> Analysis::durationList;
std::vector<std::pair<int, const char*>> Analysis::labelList;