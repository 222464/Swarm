#include "LayerConv.h"

using namespace swarm;

void LayerConv::convolve(const Int3 &pos, std::mt19937 &rng, const FloatBuffer &inputStates) {
    Int3 stateSize = getStateSize();

    int paramStartIndex = _paramsPerMap * pos.z;

    float activation = _parameters[paramStartIndex + _paramsPerMap - 1]; // Bias
    float count = 1.0f;

    for (int dx = -_spatial._filterRadius; dx <= _spatial._filterRadius; dx++)
        for (int dy = -_spatial._filterRadius; dy <= _spatial._filterRadius; dy++) {
            Int2 dPos = Int2(pos.x * _spatial._stride + dx, pos.y * _spatial._stride + dy);
            
            if (inBounds0(dPos, Int2(_inputSize.x, _inputSize.y))) {
                for (int z = 0; z < _inputSize.z; z++) {
                    int wi = paramStartIndex + (dx + _spatial._filterRadius) + (dy + _spatial._filterRadius) * _spatial._filterDiam + z * _spatial._filterArea;

                    activation += _parameters[wi] * inputStates[address3(Int3(dPos.x, dPos.y, z), Int2(_inputSize.x, _inputSize.y))];
                }

                count += _inputSize.z;
            }
        }

    if (_recurrent._filterRadius >= 0) {
        int recurrentParamStartIndex = paramStartIndex + _spatial._filterArea * _inputSize.z;

        for (int dx = -_recurrent._filterRadius; dx <= _recurrent._filterRadius; dx++)
            for (int dy = -_recurrent._filterRadius; dy <= _recurrent._filterRadius; dy++) {
                Int2 dPos = Int2(pos.x * _recurrent._stride + dx, pos.y * _recurrent._stride + dy);
                
                if (inBounds0(dPos, Int2(stateSize.x, stateSize.y))) {
                    for (int z = 0; z < _numMaps; z++) {
                        int wi = recurrentParamStartIndex + (dx + _recurrent._filterRadius) + (dy + _recurrent._filterRadius) * _recurrent._filterDiam + z * _recurrent._filterArea;

                        activation += _parameters[wi] * _statesPrev[address3(Int3(dPos.x, dPos.y, z), Int2(stateSize.x, stateSize.y))];
                    }

                    count += _numMaps;
                }
            }
    }
    
    int stateIndex = address3(pos, Int2(stateSize.x, stateSize.y));

    _states[stateIndex] = std::tanh(activation / count * _actScalar);
}

void LayerConv::create(ComputeSystem &cs, const Int3 &inputSize, int numMaps, int spatialFilterRadius, int spatialFilterStride, int recurrentFilterRadius, int recurrentFilterStride) {
    _inputSize = inputSize;
    _numMaps = numMaps;

    _spatial._filterRadius = spatialFilterRadius;
    _spatial._stride = spatialFilterStride;

    _spatial._filterDiam = _spatial._filterRadius * 2 + 1;
    _spatial._filterArea = _spatial._filterDiam * _spatial._filterDiam;

    _states.resize(_inputSize.x * _inputSize.y * _numMaps, 0.0f);

    if (recurrentFilterRadius < 0) {
        _recurrent._filterRadius = -1;
        _recurrent._stride = 0;

        _recurrent._filterDiam = 0;
        _recurrent._filterArea = 0;

        _statesPrev.clear();
    }
    else {
        _recurrent._filterRadius = recurrentFilterRadius;
        _recurrent._stride = recurrentFilterStride;

        _recurrent._filterDiam = _recurrent._filterRadius * 2 + 1;
        _recurrent._filterArea = _recurrent._filterDiam * _recurrent._filterDiam;

        _statesPrev = _states;
    }

    _paramsPerMap = _spatial._filterArea * _inputSize.z + _recurrent._filterArea * _numMaps + 1; // +1 for bias

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

    if (_recurrent._filterRadius >= 0)
        _statesPrev = _states;
}