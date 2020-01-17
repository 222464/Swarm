#include "LayerPool.h"

using namespace swarm;

void LayerPool::pool(const Int3 &pos, std::mt19937 &rng, const FloatBuffer &inputStates) {
    float maxValue = -1.0f;

    // Max pooling
    for (int dx = 0; dx < poolDiv; dx++)
        for (int dy = 0; dy < poolDiv; dy++) {
            Int2 dPos = Int2(pos.x * poolDiv + dx, pos.y * poolDiv + dy);

            if (inBounds0(dPos, Int2(inputSize.x, inputSize.y))) {
                float value = inputStates[address3(Int3(dPos.x, dPos.y, pos.z), inputSize)];

                maxValue = std::max(maxValue, value);
            }
        }

    states[address3(pos, stateSize)] = maxValue;
}

void LayerPool::create(ComputeSystem &cs, const Int3 &inputSize, int poolDiv) {
    this->inputSize = inputSize;
    this->poolDiv = poolDiv;

    stateSize = Int3(inputSize.x / poolDiv, inputSize.y / poolDiv, inputSize.z);

    states.resize(stateSize.x * stateSize.y * stateSize.z, 0.0f);
}

void LayerPool::activate(ComputeSystem &cs, const FloatBuffer &inputStates) {
    // Convolve
#ifdef KERNEL_NO_THREAD
    for (int x = 0; x < stateSize.x; x++)
        for (int y = 0; y < stateSize.y; y++)
            for (int z = 0; z < stateSize.z; z++)
                pool(Int3(x, y, z), cs.rng, inputStates);
#else
    runKernel3(cs, std::bind(LayerPool::poolKernel, std::placeholders::_1, std::placeholders::_2, this, inputStates), stateSize, cs.rng, cs.batchSize3);
#endif
}