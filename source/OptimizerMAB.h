#pragma once

#include "Optimizer.h"

namespace swarm {
    // Multi-armed bandit (MAB) optimizer
    class OptimizerMAB : public Optimizer {
    private:
        std::vector<FloatBuffer> _values;
        std::vector<IntBuffer> _indices;

        // Number of arms
        int _numArms;

        // Kernels
        void step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, float reward);

        static void stepKernel(int pos, std::mt19937 &rng, OptimizerMAB* p, int layerIndex, FloatBuffer* parameters, float reward) {
            p->step(pos, rng, layerIndex, parameters, reward);
        }

    public:
        // Average decay
        float _alpha;

        // Exploration amount
        float _epsilon;

        OptimizerMAB()
        : _alpha(0.001f), _epsilon(0.3f)
        {}

        void create(ComputeSystem &cs, const std::vector<int> &numParameters, int numArms);

        void optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, float reward) override;
    };
}
