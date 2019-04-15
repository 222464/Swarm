#pragma once

#include "Layer.h"

namespace swarm {
    // 2D convolution layer
    class LayerConv : public Layer {
    public:
        struct FilterDesc {
            int _filterRadius;
            int _stride;
            int _filterDiam;
            int _filterArea;
        };

    private:
        Int3 _inputSize;
        int _numMaps;

        FilterDesc _spatial;
        FilterDesc _recurrent;
        
        int _paramsPerMap;

        FloatBuffer _parameters;

        // Kernels
        void convolve(const Int3 &pos, std::mt19937 &rng, const FloatBuffer &inputStates);

        static void convolveKernel(const Int3 &pos, std::mt19937 &rng, LayerConv* p, const FloatBuffer &inputStates) {
            p->convolve(pos, rng, inputStates);
        }

    public:
        // Activation scalar (how quickly activation function saturates)
        float _actScalar;

        LayerConv()
        : _actScalar(8.0f)
        {}

        void create(ComputeSystem &cs, const Int3 &inputSize, int numMaps, int spatialFilterRadius, int spatialFilterStride, int recurrentFilterRadius, int recurrentFilterStride);

        void activate(ComputeSystem &cs, const FloatBuffer &inputStates) override;
        
        std::shared_ptr<Layer> clone() const override {
            return std::static_pointer_cast<Layer>(std::make_shared<LayerConv>(*this));
        }

        Int3 getStateSize() const override {
            return Int3(_inputSize.x / _spatial._stride, _inputSize.y / _spatial._stride, _numMaps);
        }

        FloatBuffer* getParameters() override {
            return &_parameters;
        }
    };
}
