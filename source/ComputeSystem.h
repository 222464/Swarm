#pragma once

#include "Helpers.h"
#include <omp.h>

#include <random>

namespace swarm {
class ComputeSystem {
public:
	// Default batch sizes for dimensions 1-3
	int batchSize1;
	Int2 batchSize2;
	Int3 batchSize3;

	// Default RNG
	std::mt19937 rng;

	ComputeSystem()
	:
	batchSize1(512),
	batchSize2(2, 2),
	batchSize3(2, 2, 2)
	{}

	static void setNumThreads(int numThreads) {
		omp_set_num_threads(numThreads);
	}

	static int getNumThreads() {
		return omp_get_num_threads();
	}
};
}