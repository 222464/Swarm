#include "LayerPool.h"

using namespace swarm;

void LayerPool::pool(const Int3 &pos, std::mt19937 &rng, const FloatBuffer &inputStates) {
    float maxValue = -1.0f; // Minimum value

    // Max pooling
    for (int dx = 0; dx < _poolDiv; dx++)
        for (int dy = 0; dy < _poolDiv; dy++) {
            Int2 dPos = Int2(pos.x * _poolDiv + dx, pos.y * _poolDiv + dy);

            if (inBounds0(dPos, Int2(_inputSize.x, _inputSize.y))) {
                float value = inputStates[address3(Int3(dPos.x, dPos.y, pos.z), Int2(_inputSize.x, _inputSize.y))];

                maxValue = std::max(maxValue, value);
            }
        }

    _states[address3(pos, Int2(_stateSize.x, _stateSize.y))] = maxValue;
}

void LayerPool::create(ComputeSystem &cs, const Int3 &inputSize, int poolDiv) {
    _inputSize = inputSize;
    _poolDiv = poolDiv;

    _stateSize = Int3(_inputSize.x / _poolDiv, _inputSize.y / _poolDiv, _inputSize.z);

    _states.resize(_stateSize.x * _stateSize.y * _stateSize.z, 0.0f);
}

void LayerPool::activate(ComputeSystem &cs, const FloatBuffer &inputStates) {
    // Convolve
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < _stateSize.x; x++)
        for (int y = 0; y < _stateSize.y; y++)
            for (int z = 0; z < _stateSize.z; z++)
                pool(Int3(x, y, z), cs._rng, inputStates);
#else
    runKernel3(cs, std::bind(LayerPool::poolKernel, std::placeholders::_1, std::placeholders::_2, this, inputStates), _stateSize, cs._rng, cs._batchSize3);
#endif
}