#pragma once

#include "Optimizer.h"

namespace swarm {
    // Multi-armed bandit (MAB) optimizer
    class OptimizerMAB : public Optimizer {
    private:
        std::vector<FloatBuffer> values;
        std::vector<IntBuffer> indices;

        // Timer for play time
        int timer;

        // Number of arms
        int numArms;

        // Kernels
        void step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, float reward, bool select);

        static void stepKernel(int pos, std::mt19937 &rng, OptimizerMAB* p, int layerIndex, FloatBuffer* parameters, float reward, bool select) {
            p->step(pos, rng, layerIndex, parameters, reward, select);
        }

    public:
        // Average decay
        float alpha;

        // Exploration overshoot
        float epsilon;

        // Ticks to try an arm
        int playTime;

        OptimizerMAB()
        : timer(0), alpha(0.01f), epsilon(0.5f), playTime(8)
        {}

        void create(ComputeSystem &cs, const std::vector<int> &numParameters, int numArms);

        void optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, float reward) override;

        int getTimer() const {
            return timer;
        }

        int getNumArms() const {
            return numArms;
        }

        std::vector<FloatBuffer> &getValues() {
            return values;
        } 

        std::vector<IntBuffer> &getIndices() {
            return indices;
        }
    };
}
