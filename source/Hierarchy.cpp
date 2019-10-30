#include "Hierarchy.h"

#include <algorithm>
#include <assert.h>

using namespace swarm;

void Hierarchy::operator=(const Hierarchy &other) {
    _layers.resize(other._layers.size());

    for (int i = 0; i < _layers.size(); i++)
        _layers[i] = other._layers[i]->clone();
}

void Hierarchy::create(const std::vector<std::shared_ptr<Layer>> &layers) {
    _layers = layers;
}

void Hierarchy::activate(ComputeSystem &cs, const FloatBuffer &inputStates) {
    // Go through layers
    for (int i = 0; i < _layers.size(); i++) {
        assert(_layers[i] != nullptr);

        _layers[i]->activate(cs, i == 0 ? inputStates : _layers[i - 1]->getStates());
    }
}

void Hierarchy::optimize(ComputeSystem &cs, Optimizer* opt, float reward) {
    std::vector<FloatBuffer*> parameters = getParameters();

    opt->optimize(cs, parameters, reward);
}

std::vector<int> Hierarchy::getNumParameters() {
    std::vector<int> numParameters(_layers.size());

    // Gather parameters
    for (int i = 0; i < _layers.size(); i++) {
        assert(_layers[i] != nullptr);

        if (_layers[i]->getParameters() == nullptr)
            numParameters[i] = 0;
        else
            numParameters[i] = _layers[i]->getParameters()->size();
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
    std::vector<FloatBuffer*> parameters(_layers.size());

    // Gather parameters
    for (int i = 0; i < _layers.size(); i++) {
        assert(_layers[i] != nullptr);

        parameters[i] = _layers[i]->getParameters();
    }

    return parameters;
}