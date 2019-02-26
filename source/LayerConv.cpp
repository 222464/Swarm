#include "LayerConv.h"

using namespace swarm;

void LayerConv::convolve(const Int3 &pos, std::mt19937 &rng, const FloatBuffer &inputStates) {
    int paramStartIndex = _paramsPerMap * pos.z;

    float activation = _parameters[paramStartIndex + _paramsPerMap - 1]; // Bias
    float count = 1.0f;

    for (int dx = -_filterRadius; dx <= _filterRadius; dx++)
        for (int dy = -_filterRadius; dy <= _filterRadius; dy++) {
            Int2 dPos = Int2(pos.x + dx, pos.y + dy);

            if (inBounds0(dPos, Int2(_inputSize.x, _inputSize.y))) {
                for (int z = 0; z < _inputSize.z; z++) {
                    int wi = paramStartIndex + (dx + _filterRadius) + (dy + _filterRadius) * _filterDiam + z * _filterArea;

                    activation += _parameters[wi] * inputStates[address3(Int3(dPos.x, dPos.y, z), Int2(_inputSize.x, _inputSize.y))];
                }

                count += _inputSize.z;

                if (_recurrent) {
                    for (int z = 0; z < _numMaps; z++) {
                        int wi = paramStartIndex + (dx + _filterRadius) + (dy + _filterRadius) * _filterDiam + (_inputSize.z + z) * _filterArea;

                        activation += _parameters[wi] * _statesPrev[address3(Int3(dPos.x, dPos.y, z), Int2(_inputSize.x, _inputSize.y))];
                    }

                    count += _numMaps;
                }
            }
        }

    _states[address3(pos, Int2(_inputSize.x, _inputSize.y))] = std::tanh(activation / count * _actScalar); // Tanh activation
}

void LayerConv::create(ComputeSystem &cs, const Int3 &inputSize, int numMaps, int filterRadius, bool recurrent) {
    _inputSize = inputSize;
    _numMaps = numMaps;

    _filterRadius = filterRadius;

    _recurrent = recurrent;

    _states.resize(_inputSize.x * _inputSize.y * _numMaps, 0.0f);

    if (_recurrent)
        _statesPrev = _states;

    _filterDiam = _filterRadius * 2 + 1;
    _filterArea = _filterDiam * _filterDiam;

    _paramsPerMap = _filterArea * (_inputSize.z + (_recurrent ? _numMaps : 0)) + 1; // +1 for bias

    _parameters.resize(_paramsPerMap * _numMaps, 0.0f);
}

void LayerConv::activate(ComputeSystem &cs, const FloatBuffer &inputStates) {
    Int3 stateSize = getStateSize();

    // Convolve
#ifdef KERNEL_NOTHREAD
    for (int x = 0; x < stateSize.x; x++)
        for (int y = 0; y < stateSize.y; y++)
            for (int z = 0; z < stateSize.z; z++)
                convolve(Int3(x, y, z), cs._rng, inputStates);
#else
    runKernel3(cs, std::bind(LayerConv::convolveKernel, std::placeholders::_1, std::placeholders::_2, this, inputStates), stateSize, cs._rng, cs._batchSize3);
#endif

    // Recurrent state
    if (_recurrent)
        _statesPrev = _states;
}