#pragma once

#include "Helpers.h"

#include <random>

namespace swarm {
	// Compute system. Mainly passed to other functions. Contains thread pooling and random number generator information
    class ComputeSystem {
	public:
		// System thread pool
		ctpl::thread_pool _pool;

		// Default batch sizes
		int _batchSize1;
		Int2 _batchSize2;
		Int3 _batchSize3;

		// Shared random generator
		std::mt19937 _rng;

		// Initialize the system
        ComputeSystem(int numWorkers)
		: _pool(numWorkers), _batchSize1(512), _batchSize2(2, 2), _batchSize3(2, 2, 4)
		{}
    };
}