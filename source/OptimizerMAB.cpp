#include "OptimizerMAB.h"

using namespace swarm;

void OptimizerMAB::step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, float reward, bool select) {
    // Trace update
    for (int i = 0; i < _numArms; i++) {
        int di = pos * _numArms + i;

        _traces[layerIndex][di] = std::max((1.0f - _beta) * _traces[layerIndex][di], _falloff[std::abs(_indices[layerIndex][pos] - i)]);
    }

    // Update previous average reward
    for (int i = 0; i < _numArms; i++) {
        int di = pos * _numArms + i;

        _values[layerIndex][di] += _alpha * _traces[layerIndex][di] * reward;//(reward - _values[layerIndex][di]);
    }

    if (select) {
        // Find new max index
        int maxIndex = 0;

        for (int i = 0; i < _numArms; i++) {
            int di = pos * _numArms + i;

            // Find max
            if (_values[layerIndex][di] > _values[layerIndex][pos * _numArms + maxIndex])
                maxIndex = i;
        }
        
        // Exploration
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

        if (dist01(rng) < _epsilon) {
            std::uniform_int_distribution<int> armDist(0, _numArms - 1);

            _indices[layerIndex][pos] = armDist(rng);
        }
        else
            _indices[layerIndex][pos] = maxIndex;

        // Set parameter/weight
        (*parameters)[pos] = (static_cast<float>(_indices[layerIndex][pos] + 1) / static_cast<float>(_numArms + 1)) * 2.0f - 1.0f;
    }
}

void OptimizerMAB::create(ComputeSystem &cs, const std::vector<int> &numParameters, int numArms) {
    _values.resize(numParameters.size());
    _traces.resize(numParameters.size());
    _indices.resize(numParameters.size());

    _numArms = numArms;

    std::uniform_real_distribution<float> armDist(-0.0001f, 0.0001f);

    for (int i = 0; i < numParameters.size(); i++) {
        if (numParameters[i] > 0) {
            _values[i].resize(numParameters[i] * _numArms);
            _traces[i].resize(_values[i].size(), 0.0f);

            _indices[i].resize(numParameters[i]);

            // Random init
            for (int j = 0; j < numParameters[i]; j++) {
                int maxIndex = 0;

                for (int k = 0; k < _numArms; k++) {
                    int index = j * _numArms + k;

                    _values[i][index] = armDist(cs._rng);

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
    bool select = _timer == 0;

    // Per-parameter optimization
    for (int i = 0; i < _indices.size(); i++) {
        if (_indices[i].empty()) {
            assert(parameters[i] == nullptr);
            
            continue;
        }

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _indices[i].size(); x++)
            step(x, cs._rng, i, parameters[i], reward, select);
#else
        runKernel1(cs, std::bind(stepKernel, std::placeholders::_1, std::placeholders::_2, this, i, parameters[i], reward, select), _indices[i].size(), cs._rng, cs._batchSize1);
#endif
    }

    if (select)
        _timer = _playTime;
    else
        _timer--;
}

void OptimizerMAB::genFalloff() {
    _falloff.resize(_numArms);

    for (int i = 0; i < _numArms; i++)
        _falloff[i] = std::exp(-_gamma * i * i);
}