#pragma once

#include "Layer.h"

namespace swarm {
    // 2D pooling layer
    class LayerPool : public Layer {
    private:
        Int3 inputSize;
        Int3 stateSize;

        // Downsampling factor
        int poolDiv;

        void pool(
            const Int3 &pos,
            std::mt19937 &rng,
            const FloatBuffer &inputStates
        );

        static void poolKernel(
            const Int3 &pos,
            std::mt19937 &rng,
            LayerPool* p,
            const FloatBuffer &inputStates
        ) {
            p->pool(pos, rng, inputStates);
        }

    public:
        // Create with given downsampling factor
        void init(
            ComputeSystem &cs,
            const Int3 &inputSize,
            int poolDiv
        );

        void activate(
            ComputeSystem &cs,
            const FloatBuffer &inputStates
        ) override;

        std::shared_ptr<Layer> clone() const override {
            return std::static_pointer_cast<Layer>(std::make_shared<LayerPool>(*this));
        }

        Int3 getStateSize() const override {
            return stateSize;
        }

        FloatBuffer* getParameters() override {
            return nullptr;
        }

        FloatBuffer* getGrads() override {
            return nullptr;
        }
    };
}
