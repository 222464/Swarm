#pragma once

#include "Layer.h"

namespace swarm {
    // 2D convolution layer
    class LayerConv : public Layer {
    public:
        struct FilterDesc {
            int _filterRadius;
            int _filterDiam;
            int _filterArea;
        };

    private:
        Int3 _inputSize;
        int _numMaps;

        FilterDesc _spatial;
        int _spatialFilterStride;
        FilterDesc _recurrent;
        
        int _paramsPerMap;

        FloatBuffer _parameters;
        FloatBuffer _grads;

        FloatBuffer _statesPrev;

        // Kernels
        void convolve(const Int3 &pos, std::mt19937 &rng, const FloatBuffer &inputStates);

        static void convolveKernel(const Int3 &pos, std::mt19937 &rng, LayerConv* p, const FloatBuffer &inputStates) {
            p->convolve(pos, rng, inputStates);
        }

    public:
        // Activation scalar (how quickly activation function saturates)
        float _actScalar;
        float _recurrentScalar;

        LayerConv()
        : _actScalar(5.0f),
        _recurrentScalar(0.5f)
        {}

        void create(ComputeSystem &cs, const Int3 &inputSize, int numMaps, int spatialFilterRadius, int spatialFilterStride, int recurrentFilterRadius);

        void activate(ComputeSystem &cs, const FloatBuffer &inputStates) override;
        
        std::shared_ptr<Layer> clone() const override {
            return std::static_pointer_cast<Layer>(std::make_shared<LayerConv>(*this));
        }

        Int3 getStateSize() const override {
            return Int3(_inputSize.x / _spatialFilterStride, _inputSize.y / _spatialFilterStride, _numMaps);
        }

        FloatBuffer* getParameters() override {
            return &_parameters;
        }

        FloatBuffer &getStatesPrev() {
            return _statesPrev;
        }
    };
}
