#pragma once

#include "Optimizer.h"

namespace swarm {
    // Dynamic optimizer
    class OptimizerDynamic : public Optimizer {
    private:
        std::vector<FloatBuffer> _ws;
        std::vector<FloatBuffer> _ts;

        // Kernels
        void step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, float reward);

        static void stepKernel(int pos, std::mt19937 &rng, OptimizerDynamic* p, int layerIndex, FloatBuffer* parameters, float reward) {
            p->step(pos, rng, layerIndex, parameters, reward);
        }

    public:
        float _alpha;
        float _gamma;
        float _epsilon;
        float _temperature;

        OptimizerDynamic()
        : _alpha(0.02f), _gamma(0.97f), _epsilon(0.1f), _temperature(0.5f)
        {}

        void create(ComputeSystem &cs, const std::vector<int> &numParameters);

        void optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, float reward) override;

        std::vector<FloatBuffer> &getWs() {
            return _ws;
        }

        std::vector<FloatBuffer> &getTs() {
            return _ts;
        }
    };
}
