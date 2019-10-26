#include "OptimizerMAB.h"

using namespace swarm;

void OptimizerMAB::step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, const FloatBuffer* grads, float reward, bool select) {
    // Update previous average reward
    int diPrev = pos * _numArms + _indices[layerIndex][pos];

    if (_committed[layerIndex][diPrev])
        _values[layerIndex][diPrev] += _alpha * (reward - _values[layerIndex][diPrev]); // Update reward

    // Commit selected
    _committed[layerIndex][diPrev] = 1;

    if (select) {
        // Find new max index
        int maxIndex = 0;
        float maxValue = -999999.0f;

        for (int i = 0; i < _numArms; i++) {
            int di = pos * _numArms + i;

            if (_committed[layerIndex][di] == 0)
                continue;

            if (_values[layerIndex][di] > maxValue) {
                maxValue = _values[layerIndex][di];

                maxIndex = i;
            }
        }
        
        // Exploration
        if (_epsilon == 0.0f)
            _indices[layerIndex][pos] = maxIndex;
        else { // Explore around index with Gaussian
            int delta = std::round((*grads)[pos] * _epsilon);

            _indices[layerIndex][pos] = std::min(_numArms - 1, std::max(0, maxIndex + delta));
        }

        // Set parameter/weight
        (*parameters)[pos] = (static_cast<float>(_indices[layerIndex][pos] + 1) / static_cast<float>(_numArms + 1)) * 2.0f - 1.0f;
    }
}

void OptimizerMAB::create(ComputeSystem &cs, const std::vector<int> &numParameters, int numArms) {
    _values.resize(numParameters.size());
    _committed.resize(numParameters.size());
    _indices.resize(numParameters.size());

    _numArms = numArms;

    for (int i = 0; i < numParameters.size(); i++) {
        if (numParameters[i] > 0) {
            _values[i].resize(numParameters[i] * _numArms, 0.0f);

            _committed[i].resize(numParameters[i] * _numArms, 0);

            _indices[i].resize(numParameters[i], _numArms / 2);
        }
    }
}

void OptimizerMAB::optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, const std::vector<FloatBuffer*> &grads, float reward) {
    bool select = _timer == 0;

    // Per-parameter optimization
    for (int i = 0; i < _indices.size(); i++) {
        if (_indices[i].empty()) {
            assert(parameters[i] == nullptr);
            
            continue;
        }

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _indices[i].size(); x++)
            step(x, cs._rng, i, parameters[i], grads[i], reward, select);
#else
        runKernel1(cs, std::bind(stepKernel, std::placeholders::_1, std::placeholders::_2, this, i, parameters[i], grads[i], reward, select), _indices[i].size(), cs._rng, cs._batchSize1);
#endif
    }

    if (select)
        _timer = _playTime;

    if (_timer > 0)
        _timer--;
}