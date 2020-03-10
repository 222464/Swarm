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
        void step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, const FloatBuffer* grads, float reward, bool select);

        static void stepKernel(int pos, std::mt19937 &rng, OptimizerMAB* p, int layerIndex, FloatBuffer* parameters, const FloatBuffer* grads, float reward, bool select) {
            p->step(pos, rng, layerIndex, parameters, grads, reward, select);
        }

    public:
        // Average decay
        float alpha;

        // Gradient scale
        float beta;

        // Exploration chance
        float epsilon;

        // Ticks to try an arm
        int playTime;

        OptimizerMAB()
        : timer(0), alpha(0.01f), beta(0.01f), epsilon(0.2f), playTime(8)
        {}

        void init(ComputeSystem &cs, const std::vector<int> &numParameters, int numArms);

        void optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, const std::vector<FloatBuffer*> &grads, float reward) override;

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
