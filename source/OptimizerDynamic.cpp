#include "OptimizerDynamic.h"
#include <iostream>
using namespace swarm;

const float _pi = 3.141596f;

float OptimizerDynamic::WFunc(float W, float A, float C, int T, int timer) {
    return A * std::sin(2.0f * _pi * static_cast<float>(timer) / static_cast<float>(T)) + C;
}

void OptimizerDynamic::step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, float reward) {
    // Reinforcement
    _Cs[layerIndex][pos] += _alpha * reward * (_Ws[layerIndex][pos] - _Cs[layerIndex][pos]);
    _As[layerIndex][pos] += -_beta * (_As[layerIndex][pos] > 0.0f ? 1.0f : -1.0f) * reward;

    // Find weight
    _Ws[layerIndex][pos] = WFunc(_Ws[layerIndex][pos], _As[layerIndex][pos], _Cs[layerIndex][pos], _Ts[layerIndex][pos], _timers[layerIndex][pos]);

    // Update timer
    _timers[layerIndex][pos]++;

    if (_timers[layerIndex][pos] >= _Ts[layerIndex][pos]) {
        _timers[layerIndex][pos] = 0;

        std::normal_distribution<float> TDist(_mu, _sigma);

        _Ts[layerIndex][pos] = std::max(1, static_cast<int>(TDist(rng) + 0.5f));
    }

    // Set parameter/weight
    (*parameters)[pos] = _Ws[layerIndex][pos];
}

void OptimizerDynamic::create(ComputeSystem &cs, const std::vector<int> &numParameters) {
    _Ws.resize(numParameters.size());
    _Cs.resize(numParameters.size());
    _As.resize(numParameters.size());
    _Ts.resize(numParameters.size());
    _timers.resize(numParameters.size());

    std::uniform_real_distribution<float> CDist(-1.0, 1.0f);
    std::normal_distribution<float> ADist(0.0f, 1.0f);
    std::normal_distribution<float> TDist(_mu, _sigma);

    for (int i = 0; i < numParameters.size(); i++) {
        if (numParameters[i] > 0) {
            _Ws[i].resize(numParameters[i], 0.0f);
            _Cs[i].resize(numParameters[i]);
            _As[i].resize(numParameters[i]);
            _Ts[i].resize(numParameters[i]);

            _timers[i].resize(numParameters[i], 0);

            // Random init
            for (int j = 0; j < numParameters[i]; j++) {
                _Cs[i][j] = CDist(cs._rng);
                _As[i][j] = ADist(cs._rng);
                _Ts[i][j] = std::max(1, static_cast<int>(TDist(cs._rng) + 0.5f));
            } 
        }
    }
}

void OptimizerDynamic::optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, float reward) {
    // Per-parameter optimization
    for (int i = 0; i < _Ws.size(); i++) {
        if (_Ws[i].empty()) {
            assert(parameters[i] == nullptr);
            
            continue;
        }

#ifdef KERNEL_NOTHREAD
        for (int x = 0; x < _Ws[i].size(); x++)
            step(x, cs._rng, i, parameters[i], reward);
#else
        runKernel1(cs, std::bind(stepKernel, std::placeholders::_1, std::placeholders::_2, this, i, parameters[i], reward), _Ws[i].size(), cs._rng, cs._batchSize1);
#endif
    }
}