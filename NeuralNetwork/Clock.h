#pragma once
#ifdef _WIN64
#include <chrono>

struct Clock {
private:
	std::chrono::high_resolution_clock::time_point start;
public:
	void Start() {
		start = std::chrono::high_resolution_clock::now();
	}
	long long ElapsedMicroseconds() {
		return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
	}
	long long ElapsedMilliseconds() {
		return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
	}
	long long ElapsedNanoseconds() {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();
	}
	long long ElapsedSeconds() {
		return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count();
	}
};
#else
#include <time.h>

struct Clock {
private:
	clock_t start;
public:
	void Start() {
		start = clock();
	}
	long long ElapsedMicroseconds() {
		return clock() - start / CLOCKS_PER_SEC * 1000000;
	}
	long long ElapsedMilliseconds() {
		return clock() - start / CLOCKS_PER_SEC * 1000;
	}
	long long ElapsedNanoseconds() {
		return clock() - start / CLOCKS_PER_SEC * 1000000000;
	}
	long long ElapsedSeconds() {
		return clock() - start / CLOCKS_PER_SEC;
	}
};
#endif