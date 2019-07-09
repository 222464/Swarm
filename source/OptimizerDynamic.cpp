#include "OptimizerDynamic.h"

using namespace swarm;

void OptimizerDynamic::step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, float reward) {
    // Reinforcement
    _ws[layerIndex][pos] += _alpha * reward * _ts[layerIndex][pos];

    // Set parameter/weight
    
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    float noise;
    
    if (dist01(rng) < _epsilon) {
        std::normal_distribution<float> noiseDist(0.0f, _temperature);

        noise = noiseDist(rng);
    }
    else
        noise = 0.0f;

    (*parameters)[pos] = _ws[layerIndex][pos] + noise;

    _ts[layerIndex][pos] = _gamma * _ts[layerIndex][pos] + noise;
}

void OptimizerDynamic::create(ComputeSystem &cs, const std::vector<int> &numParameters) {
    _ws.resize(numParameters.size());
    _ts.resize(numParameters.size());

    std::uniform_real_distribution<float> wDist(-0.001f, 0.001f);

    for (int i = 0; i < numParameters.size(); i++) {
        if (numParameters[i] > 0) {
            _ws[i].resize(numParameters[i]);
            _ts[i].resize(numParameters[i], 0.0f);

            // Random init
            for (int j = 0; j < numParameters[i]; j++)
                _ws[i][j] = wDist(cs._rng);
        }
    }
}

void OptimizerDynamic::optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, float reward) {
    // Per-parameter optimization
    for (int i = 0; i < _ws.size(); i++) {
        if (_ws[i].empty()) {
            assert(parameters[i] == nullptr);
            
            continue;
        }

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _ws[i].size(); x++)
            step(x, cs._rng, i, parameters[i], reward);
#else
        runKernel1(cs, std::bind(stepKernel, std::placeholders::_1, std::placeholders::_2, this, i, parameters[i], reward), _ws[i].size(), cs._rng, cs._batchSize1);
#endif
    }
}