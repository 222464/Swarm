#pragma once

#include "ComputeSystem.h"

namespace swarm {
    // Base class for layers
    class Layer {
    protected:
        FloatBuffer states;
        
    public:
        virtual ~Layer() {}

        virtual void activate(
            ComputeSystem &cs,
            const FloatBuffer &inputStates
        ) = 0;

        // Hidden state size
        virtual Int3 getStateSize() const = 0;

        virtual FloatBuffer* getParameters() = 0;
        virtual FloatBuffer* getGrads() = 0;

        // Duplication factory
        virtual std::shared_ptr<Layer> clone() const = 0;

        FloatBuffer &getStates() {
            return states;
        }
    };
}
