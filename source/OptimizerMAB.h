#pragma once

#include "Optimizer.h"

namespace swarm {
    // Multi-armed bandit (MAB) optimizer
    class OptimizerMAB : public Optimizer {
    private:
        std::vector<FloatBuffer> _values;
        std::vector<IntBuffer> _indices;

        // Timer for play time
        int _timer;

        // Number of arms
        int _numArms;

        // Kernels
        void step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, float reward, bool select);

        static void stepKernel(int pos, std::mt19937 &rng, OptimizerMAB* p, int layerIndex, FloatBuffer* parameters, float reward, bool select) {
            p->step(pos, rng, layerIndex, parameters, reward, select);
        }

    public:
        // Average decay
        float _alpha;

        // Exploration amount
        float _epsilon;

        // Ticks to try an arm
        int _playTime;

        OptimizerMAB()
        : _timer(0), _alpha(0.01f), _epsilon(0.5f), _playTime(16)
        {}

        void create(ComputeSystem &cs, const std::vector<int> &numParameters, int numArms);

        void optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, float reward) override;
    };
}
