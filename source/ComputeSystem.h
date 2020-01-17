#pragma once

#include "Helpers.h"

#include <random>

namespace swarm {
	// Compute system. Mainly passed to other functions. Contains thread pooling and random number generator information
    class ComputeSystem {
	public:
		// System thread pool
		ctpl::thread_pool pool;

		// Default batch sizes
		int batchSize1;
		Int2 batchSize2;
		Int3 batchSize3;

		// Shared random generator
		std::mt19937 rng;

		// Initialize the system
        ComputeSystem(int numWorkers)
		: pool(numWorkers), batchSize1(1024), batchSize2(1, 1), batchSize3(1, 1, 1)
		{}
    };
}