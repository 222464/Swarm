#include "OptimizerMAB.h"

using namespace swarm;

void OptimizerMAB::step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, float reward) {
    // Update previous average reward
    int diPrev = pos * _numArms + _indices[layerIndex][pos];

    _dists[layerIndex][diPrev] += _alpha * (reward - _dists[layerIndex][diPrev]);

    // Find new max index
    int maxIndex = 0;

    for (int i = 0; i < _numArms; i++) {
        int di = pos * _numArms + i;

        if (_dists[layerIndex][di] > _dists[layerIndex][pos * _numArms + maxIndex])
            maxIndex = i;
    }
    
    // Exploration
    if (_epsilon == 0.0f)
        _indices[layerIndex][pos] = maxIndex;
    else { // Explore around index with Gaussian
        std::normal_distribution<float> noiseDist(0.0f, _epsilon);

        _indices[layerIndex][pos] = std::min(_numArms - 1, std::max(0, static_cast<int>(maxIndex + 0.5f + noiseDist(rng))));
    }

    // Set parameter/weight
    (*parameters)[pos] = logit(static_cast<float>(_indices[layerIndex][pos] + 1) / static_cast<float>(_numArms + 1));
}

void OptimizerMAB::create(ComputeSystem &cs, const std::vector<int> &numParameters, int numArms) {
    _dists.resize(numParameters.size());
    _indices.resize(numParameters.size());

    _numArms = numArms;

    std::uniform_real_distribution<float> distDist(-0.0001f, 0.0001f);

    for (int i = 0; i < numParameters.size(); i++) {
        if (numParameters[i] > 0) {
            _dists[i].resize(numParameters[i] * _numArms);

            _indices[i].resize(numParameters[i]);

            // Random init
            for (int j = 0; j < numParameters[i]; j++) {
                int maxIndex = 0;

                for (int k = 0; k < _numArms; k++) {
                    int index = j * _numArms + k;

                    _dists[i][index] = distDist(cs._rng);

                    if (_dists[i][index] > _dists[i][j * _numArms + maxIndex])
                        maxIndex = k;
                }

                _indices[i][j] = maxIndex;
            } 
        }
    }
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