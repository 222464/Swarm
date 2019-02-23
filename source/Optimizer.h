#pragma once

#include "ComputeSystem.h"

namespace swarm {
    // Base class for optimizers
    class Optimizer {
    public:
        virtual ~Optimizer() {}

        virtual void optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, float reward) = 0;
    };
}
