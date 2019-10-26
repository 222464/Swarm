#include "OptimizerMAB.h"

using namespace swarm;

void OptimizerMAB::step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, float reward, bool select) {
    // Update previous average reward
    int diPrev = pos * _numArms + _indices[layerIndex][pos];

    _values[layerIndex][diPrev] += _alpha * (reward - _values[layerIndex][diPrev]); // Update reward

    if (select) {
        // Find new max index
        int maxIndex = 0;
        float maxValue = -999999.0f;

        for (int i = 0; i < _numArms; i++) {
            int di = pos * _numArms + i;

            if (_values[layerIndex][di] > maxValue) {
                maxValue = _values[layerIndex][di];

                maxIndex = i;
            }
        }
        
        // Exploration
        if (_epsilon == 0.0f)
            _indices[layerIndex][pos] = maxIndex;
        else { // Explore around index with Gaussian
            std::normal_distribution<float> normDist(0.0f, 1.0f);

            int delta = std::round(normDist(rng) * _epsilon);

            _indices[layerIndex][pos] = std::min(_numArms - 1, std::max(0, maxIndex + delta));
        }

        // Set parameter/weight
        (*parameters)[pos] = (static_cast<float>(_indices[layerIndex][pos] + 1) / static_cast<float>(_numArms + 1)) * 2.0f - 1.0f;
    }
}

void OptimizerMAB::create(ComputeSystem &cs, const std::vector<int> &numParameters, int numArms) {
    _values.resize(numParameters.size());
    _indices.resize(numParameters.size());

    _numArms = numArms;

    std::uniform_real_distribution<float> noiseDist(-0.001f, 0.001f);

    for (int i = 0; i < numParameters.size(); i++) {
        if (numParameters[i] > 0) {
            _values[i].resize(numParameters[i] * _numArms);

            for (int j = 0; j < _values[i].size(); j++)
                _values[i][j] = noiseDist(cs._rng);

            _indices[i].resize(numParameters[i], _numArms / 2);
        }
    }
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

    if (_timer > 0)
        _timer--;
}