#pragma once

#include "Layer.h"
#include "Optimizer.h"
#include "Helpers.h"

#include <memory>

namespace swarm {
    // A hierarchy of layers (layers are maintained separately)
    class Hierarchy {
    private:
        std::vector<std::shared_ptr<Layer>> layers;

    public:
        Hierarchy() {}
        Hierarchy(const Hierarchy &other) {
            *this = other;
        }

        void operator=(const Hierarchy &other);

        void create(const std::vector<std::shared_ptr<Layer>> &layers);

        // Activate off of input states
        void activate(ComputeSystem &cs, const FloatBuffer &inputStates);

        // Optimize with given optimizer and reward
        void optimize(ComputeSystem &cs, Optimizer* opt, float reward);

        std::vector<int> getNumParameters();
        int getTotalNumParameters();
        std::vector<FloatBuffer*> getParameters();

        std::vector<std::shared_ptr<Layer>> &getLayers() {
            return layers;
        }
    };
}
