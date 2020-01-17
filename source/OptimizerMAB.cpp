#include "OptimizerMAB.h"

using namespace swarm;

void OptimizerMAB::step(int pos, std::mt19937 &rng, int layerIndex, FloatBuffer* parameters, float reward, bool select) {
    // Update previous average reward
    int diPrev = pos * numArms + indices[layerIndex][pos];

    values[layerIndex][diPrev] += alpha * (reward - values[layerIndex][diPrev]); // Update reward

    if (select) {
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
        
        std::normal_distribution<float> distNorm(0.0f, 1.0f);
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

        float delta = maxIndex + distNorm(rng) * epsilon - indices[layerIndex][pos];

        indices[layerIndex][pos] = std::min(numArms - 1, std::max(0, static_cast<int>(std::round(indices[layerIndex][pos] + dist01(rng) * delta))));

        // Set parameter/weight
        (*parameters)[pos] = (static_cast<float>(indices[layerIndex][pos] + 1) / static_cast<float>(numArms + 1)) * 2.0f - 1.0f;
    }
}

void OptimizerMAB::create(ComputeSystem &cs, const std::vector<int> &numParameters, int numArms) {
    values.resize(numParameters.size());
    indices.resize(numParameters.size());

    this->numArms = numArms;

    std::uniform_real_distribution<float> noiseDist(-0.001f, 0.001f);

    for (int i = 0; i < numParameters.size(); i++) {
        if (numParameters[i] > 0) {
            values[i].resize(numParameters[i] * numArms);

            for (int j = 0; j < values[i].size(); j++)
                values[i][j] = noiseDist(cs.rng);

            indices[i].resize(numParameters[i], numArms / 2);
        }
    }
}

void OptimizerMAB::optimize(ComputeSystem &cs, std::vector<FloatBuffer*> &parameters, float reward) {
    bool select = timer == 0;

    // Per-parameter optimization
    for (int i = 0; i < indices.size(); i++) {
        if (indices[i].empty()) {
            assert(parameters[i] == nullptr);
            
            continue;
        }

#ifdef KERNEL_NO_THREAD
        for (int x = 0; x < indices[i].size(); x++)
            step(x, cs.rng, i, parameters[i], reward, select);
#else
        runKernel1(cs, std::bind(stepKernel, std::placeholders::_1, std::placeholders::_2, this, i, parameters[i], reward, select), indices[i].size(), cs.rng, cs.batchSize1);
#endif
    }

    if (select)
        timer = playTime;

    if (timer > 0)
        timer--;
}