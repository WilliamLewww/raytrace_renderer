#pragma once
#include <iostream>
#include <chrono>

class Analysis {
private:
	static std::chrono::high_resolution_clock::time_point start;
	static std::chrono::high_resolution_clock::time_point finish;
public:
	inline static void begin() { start = std::chrono::high_resolution_clock::now(); }
	inline static void end() { finish = std::chrono::high_resolution_clock::now(); }

	inline static void print() {
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count();
    	std::cout << duration << std::endl;
	}
};

std::chrono::high_resolution_clock::time_point Analysis::start;
std::chrono::high_resolution_clock::time_point Analysis::finish;