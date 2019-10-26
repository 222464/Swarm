#pragma once

#include "Optimizer.h"

namespace swarm {
    // Multi-armed bandit (MAB) optimizer
    class OptimizerMAB : public Optimizer {
    private:
        std::vector<FloatBuffer> _values;
        std::vector<IntBuffer> _indices;
        FloatBuffer _falloff;

        // Timer for play time
        int _timer;

        // Number of arms
        int _numArms;

        // Kernels
        void step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, const FloatBuffer* grads, float reward, bool select);

        static void stepKernel(int pos, std::mt19937 &rng, OptimizerMAB* p, int layerIndex, FloatBuffer* parameters, const FloatBuffer* grads, float reward, bool select) {
            p->step(pos, rng, layerIndex, parameters, grads, reward, select);
        }

    public:
        // Average decay
        float _alpha;

        // Falloff strength
        float _gamma;

        // Exploration amount
        float _epsilon;

        // Ticks to try an arm
        int _playTime;

        OptimizerMAB()
        : _timer(0), _alpha(0.001f), _gamma(0.03f), _epsilon(0.7f), _playTime(8)
        {}

        void create(ComputeSystem &cs, const std::vector<int> &numParameters, int numArms);

        void optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, const std::vector<FloatBuffer*> &grads, float reward) override;

        void genFalloff();

        int getTimer() const {
            return _timer;
        }

        int getNumArms() const {
            return _numArms;
        }

        std::vector<FloatBuffer> &getValues() {
            return _values;
        } 

        std::vector<IntBuffer> &getIndices() {
            return _indices;
        }
    };
}
