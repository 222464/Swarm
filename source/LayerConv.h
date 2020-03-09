#pragma once

#include "Layer.h"

namespace swarm {
    // 2D convolution layer
    class LayerConv : public Layer {
    public:
        struct FilterDesc {
            int filterRadius;
            int filterDiam;
            int filterArea;
        };

    private:
        Int3 inputSize;
        int numMaps;

        FilterDesc spatial;
        int spatialFilterStride;
        FilterDesc recurrent;
        
        int paramsPerMap;

        FloatBuffer parameters;
        FloatBuffer grads;

        FloatBuffer statesPrev;

        // Kernels
        void convolve(const Int3 &pos, std::mt19937 &rng, const FloatBuffer &inputStates);

        static void convolveKernel(const Int3 &pos, std::mt19937 &rng, LayerConv* p, const FloatBuffer &inputStates) {
            p->convolve(pos, rng, inputStates);
        }

    public:
        // Activation scalar (how quickly activation function saturates)
        float actScalar;
        float recurrentScalar;

        LayerConv()
        : actScalar(4.0f),
        recurrentScalar(0.5f)
        {}

        void init(ComputeSystem &cs, const Int3 &inputSize, int numMaps, int spatialFilterRadius, int spatialFilterStride, int recurrentFilterRadius);

        void activate(ComputeSystem &cs, const FloatBuffer &inputStates) override;
        
        std::shared_ptr<Layer> clone() const override {
            return std::static_pointer_cast<Layer>(std::make_shared<LayerConv>(*this));
        }

        Int3 getStateSize() const override {
            return Int3(inputSize.x / spatialFilterStride, inputSize.y / spatialFilterStride, numMaps);
        }

        FloatBuffer* getParameters() override {
            return &parameters;
        }

        FloatBuffer* getGrads() override {
            return &grads;
        }

        FloatBuffer &getStatesPrev() {
            return statesPrev;
        }
    };
}
