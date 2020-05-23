#include "OptimizerMAB.h"

using namespace swarm;

void OptimizerMAB::step(
    int pos,
    std::mt19937 &rng,
    int layerIndex,
    FloatBuffer* parameters,
    const FloatBuffer* grads,
    float reward,
    bool select
) {
    // Update previous average reward
    int diPrev = pos * numArms + indices[layerIndex][pos];

    values[layerIndex][diPrev] += alpha * (reward - values[layerIndex][diPrev]); // Update reward

    if (select) {
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

        if (dist01(rng) < epsilon)
            indices[layerIndex][pos] = std::min(numArms - 1, std::max(0, indices[layerIndex][pos] + ((*grads)[pos] > 0.0f ? 1 : -1)));
        else {
            // Find new max index
            int maxIndex = 0;
            float maxValue = -999999.0f;

            for (int i = 0; i < numArms; i++) {
                int di = pos * numArms + i;

                if (values[layerIndex][di] > maxValue) {
                    maxValue = values[layerIndex][di];

                    maxIndex = i;
                }
            }

            indices[layerIndex][pos] = maxIndex;
        }

        // Set parameter/weight
        (*parameters)[pos] = (static_cast<float>(indices[layerIndex][pos] + 1) / static_cast<float>(numArms + 1)) * 2.0f - 1.0f;
    }
}

void OptimizerMAB::init(
    ComputeSystem &cs,
    const std::vector<int> &numParameters,
    int numArms
) {
    values.resize(numParameters.size());
    indices.resize(numParameters.size());

    this->numArms = numArms;

    std::uniform_real_distribution<float> noiseDist(-0.001f, 0.001f);
    std::uniform_int_distribution<int> armDist(0, numArms - 1);

    for (int i = 0; i < numParameters.size(); i++) {
        if (numParameters[i] > 0) {
            values[i].resize(numParameters[i] * numArms);

            for (int j = 0; j < values[i].size(); j++)
                values[i][j] = noiseDist(cs.rng);

            indices[i].resize(numParameters[i], armDist(cs.rng));
        }
    }
}

void OptimizerMAB::optimize(
    ComputeSystem &cs,
    std::vector<FloatBuffer*> &parameters,
    const std::vector<FloatBuffer*> &grads,
    float reward
) {
    bool select = timer == 0;

    // Per-parameter optimization
    for (int i = 0; i < indices.size(); i++) {
        if (indices[i].empty()) {
            assert(parameters[i] == nullptr);
            
            continue;
        }

        runKernel1(cs, std::bind(stepKernel, std::placeholders::_1, std::placeholders::_2, this, i, parameters[i], grads[i], reward, select), indices[i].size(), cs.rng, cs.batchSize1);
    }

    if (select)
        timer = playTime;

    if (timer > 0)
        timer--;
}