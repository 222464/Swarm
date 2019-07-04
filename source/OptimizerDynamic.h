#pragma once

#include "Optimizer.h"

namespace swarm {
    // Dynamic optimizer
    class OptimizerDynamic : public Optimizer {
    private:
        std::vector<FloatBuffer> _Ws;
        std::vector<FloatBuffer> _As;
        std::vector<FloatBuffer> _Cs;
        std::vector<IntBuffer> _Ts;
        std::vector<IntBuffer> _timers;

        // Kernels
        void step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, float reward);

        static void stepKernel(int pos, std::mt19937 &rng, OptimizerDynamic* p, int layerIndex, FloatBuffer* parameters, float reward) {
            p->step(pos, rng, layerIndex, parameters, reward);
        }

        static float WFunc(float W, float A, float C, int T, int timer);

    public:
        float _alpha;
        float _beta;

        float _mu;
        float _sigma;

        OptimizerDynamic()
        : _alpha(0.001f), _beta(0.001f), _mu(100.0f), _sigma(30.0f)
        {}

        void create(ComputeSystem &cs, const std::vector<int> &numParameters);

        void optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, float reward) override;

        std::vector<FloatBuffer> &getWs() {
            return _Ws;
        }

        std::vector<FloatBuffer> &getAs() {
            return _As;
        }

        std::vector<FloatBuffer> &getCs() {
            return _Cs;
        }

        std::vector<IntBuffer> &getTs() {
            return _Ts;
        }

        std::vector<IntBuffer> &getTimers() {
            return _timers;
        }
    };
}
