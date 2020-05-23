#include "Hierarchy.h"

#include <algorithm>
#include <assert.h>

using namespace swarm;

void Hierarchy::operator=(
    const Hierarchy &other
) {
    layers.resize(other.layers.size());

    for (int i = 0; i < layers.size(); i++)
        layers[i] = other.layers[i]->clone();
}

void Hierarchy::init(
    const std::vector<std::shared_ptr<Layer>> &layers
) {
    this->layers = layers;
}

void Hierarchy::activate(
    ComputeSystem &cs,
    const FloatBuffer &inputStates
) {
    // Go through layers
    for (int i = 0; i < layers.size(); i++) {
        assert(layers[i] != nullptr);

        layers[i]->activate(cs, i == 0 ? inputStates : layers[i - 1]->getStates());
    }
}

void Hierarchy::optimize(
    ComputeSystem &cs,
    Optimizer* opt,
    float reward
) {
    std::vector<FloatBuffer*> parameters = getParameters();
    std::vector<FloatBuffer*> grads = getGrads();

    opt->optimize(cs, parameters, grads, reward);
}

std::vector<int> Hierarchy::getNumParameters() {
    std::vector<int> numParameters(layers.size());

    // Gather parameters
    for (int i = 0; i < layers.size(); i++) {
        assert(layers[i] != nullptr);

        if (layers[i]->getParameters() == nullptr)
            numParameters[i] = 0;
        else
            numParameters[i] = layers[i]->getParameters()->size();
    }

    return numParameters;
}

int Hierarchy::getTotalNumParameters() {
    std::vector<int> numParameters = getNumParameters();

    int total = 0;

    for (int i = 0; i < numParameters.size(); i++)
        total += numParameters[i];

    return total;
}

std::vector<FloatBuffer*> Hierarchy::getParameters() {
    std::vector<FloatBuffer*> parameters(layers.size());

    // Gather parameters
    for (int i = 0; i < layers.size(); i++) {
        assert(layers[i] != nullptr);

        parameters[i] = layers[i]->getParameters();
    }

    return parameters;
}

std::vector<FloatBuffer*> Hierarchy::getGrads() {
    std::vector<FloatBuffer*> grads(layers.size());

    // Gather grads
    for (int i = 0; i < layers.size(); i++) {
        assert(layers[i] != nullptr);

        grads[i] = layers[i]->getGrads();
    }

    return grads;
}