#pragma once
#include <iostream>
#include <chrono>
#include <vector>

class Analysis {
private:
	static std::chrono::high_resolution_clock::time_point start;
	static std::chrono::high_resolution_clock::time_point finish;

	static std::vector<std::vector<int64_t>> durationList;
public:
	inline static void begin() { start = std::chrono::high_resolution_clock::now(); }
	inline static void end() { finish = std::chrono::high_resolution_clock::now(); }

	inline static void print() {
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    	std::cout << duration << std::endl;
	}

	inline static void appendDuration(int index) {
		if (index >= durationList.size()) {
			durationList.push_back(std::vector<int64_t>());
		}

		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
		durationList[index].push_back(duration);
	}

	inline static void printAll() {
		std::cout << std::endl;

		int64_t averageT = 0;
		for (int x = 0; x < durationList.size(); x++) {
			int64_t average = 0;

			for (int y = 0; y < durationList[x].size(); y++) {
				average += durationList[x][y];
			}

			std::cout << "[" << x << "]: ";
			std::cout << average / durationList[x].size() << std::endl;

			averageT += average / durationList[x].size();
		}

		std::cout << "Total: " << averageT << std::endl;
		std::cout << std::endl;
	}
};

std::chrono::high_resolution_clock::time_point Analysis::start;
std::chrono::high_resolution_clock::time_point Analysis::finish;

std::vector<std::vector<int64_t>> Analysis::durationList;