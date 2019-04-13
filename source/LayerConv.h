#pragma once

#include "Layer.h"

namespace swarm {
    // 2D convolution layer
    class LayerConv : public Layer {
    private:
        Int3 _inputSize;
        int _numMaps;

        int _filterRadius;
        int _stride;
        int _filterDiam;
        int _filterArea;
        int _paramsPerMap;

        // Whether there are recurrent connections
        bool _recurrent;

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
        : _actScalar(4.0f)
        {}

        void create(ComputeSystem &cs, const Int3 &inputSize, int numMaps, int filterRadius, int stride, bool recurrent);

        void activate(ComputeSystem &cs, const FloatBuffer &inputStates) override;
        
        std::shared_ptr<Layer> clone() const override {
            return std::static_pointer_cast<Layer>(std::make_shared<LayerConv>(*this));
        }

        Int3 getStateSize() const override {
            return Int3(_inputSize.x / _stride, _inputSize.y / _stride, _numMaps);
        }

        FloatBuffer* getParameters() override {
            return &_parameters;
        }
    };
}
