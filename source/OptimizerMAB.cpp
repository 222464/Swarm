#include "OptimizerMAB.h"

using namespace swarm;

void OptimizerMAB::step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, float reward) {
    // Find new max index
    int maxIndex = 0;

    for (int i = 0; i < _numArms; i++) {
        int di = pos * _numArms + i;

        float strength = _falloff[std::abs(_indices[layerIndex][pos] - i)];

        _values[layerIndex][di] += _alpha * strength * (reward - _values[layerIndex][di]);

        // Find max
        if (_values[layerIndex][di] > _values[layerIndex][pos * _numArms + maxIndex])
            maxIndex = i;
    }
    
    _indices[layerIndex][pos] = maxIndex;
    
    // Set parameter/weight
    (*parameters)[pos] = (static_cast<float>(_indices[layerIndex][pos] + 1) / static_cast<float>(_numArms + 1)) * 2.0f - 1.0f;
}

void OptimizerMAB::create(ComputeSystem &cs, const std::vector<int> &numParameters, int numArms) {
    _values.resize(numParameters.size());
    _indices.resize(numParameters.size());

    _numArms = numArms;

    std::uniform_real_distribution<float> distDist(-0.0001f, 0.0001f);

    for (int i = 0; i < numParameters.size(); i++) {
        if (numParameters[i] > 0) {
            _values[i].resize(numParameters[i] * _numArms);

            _indices[i].resize(numParameters[i]);

            // Random init
            for (int j = 0; j < numParameters[i]; j++) {
                int maxIndex = 0;

                for (int k = 0; k < _numArms; k++) {
                    int index = j * _numArms + k;

                    _values[i][index] = distDist(cs._rng);

                    if (_values[i][index] > _values[i][j * _numArms + maxIndex])
                        maxIndex = k;
                }

                _indices[i][j] = maxIndex;
            } 
        }
    }

    genFalloff();
}

void OptimizerMAB::optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, float reward) {
    // Per-parameter optimization
    for (int i = 0; i < _indices.size(); i++) {
        if (_indices[i].empty()) {
            assert(parameters[i] == nullptr);
            
            continue;
        }

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _indices[i].size(); x++)
            step(x, cs._rng, i, parameters[i], reward);
#else
        runKernel1(cs, std::bind(stepKernel, std::placeholders::_1, std::placeholders::_2, this, i, parameters[i], reward), _indices[i].size(), cs._rng, cs._batchSize1);
#endif
    }
}

void OptimizerMAB::genFalloff() {
    _falloff.resize(_numArms);

    for (int i = 0; i < _numArms; i++)
        _falloff[i] = std::exp(-_gamma * i * i);
}