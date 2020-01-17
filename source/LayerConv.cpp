#include "LayerConv.h"

using namespace swarm;

void LayerConv::convolve(const Int3 &pos, std::mt19937 &rng, const FloatBuffer &inputStates) {
    Int3 stateSize = getStateSize();

    int paramStartIndex = paramsPerMap * pos.z;

    float activation = parameters[paramStartIndex + paramsPerMap - 1]; // Bias
    int count = 1;

    for (int dx = -spatial.filterRadius; dx <= spatial.filterRadius; dx++)
        for (int dy = -spatial.filterRadius; dy <= spatial.filterRadius; dy++) {
            Int2 dPos = Int2(pos.x * spatialFilterStride + dx, pos.y * spatialFilterStride + dy);
            
            if (inBounds0(dPos, Int2(inputSize.x, inputSize.y))) {
                for (int z = 0; z < inputSize.z; z++) {
                    int wi = paramStartIndex + (dx + spatial.filterRadius) + (dy + spatial.filterRadius) * spatial.filterDiam + z * spatial.filterArea;

                    activation += parameters[wi] * inputStates[address3(Int3(dPos.x, dPos.y, z), inputSize)];
                }

                count += inputSize.z;
            }
        }

    if (recurrent.filterRadius >= 0) {
        float recurrentActivation = 0.0f;

        int recurrentParamStartIndex = paramStartIndex + spatial.filterArea * inputSize.z;

        for (int dx = -recurrent.filterRadius; dx <= recurrent.filterRadius; dx++)
            for (int dy = -recurrent.filterRadius; dy <= recurrent.filterRadius; dy++) {
                Int2 dPos = Int2(pos.x + dx, pos.y + dy);
                
                if (inBounds0(dPos, Int2(stateSize.x, stateSize.y))) {
                    for (int z = 0; z < numMaps; z++) {
                        int wi = recurrentParamStartIndex + (dx + recurrent.filterRadius) + (dy + recurrent.filterRadius) * recurrent.filterDiam + z * recurrent.filterArea;

                        recurrentActivation += parameters[wi] * statesPrev[address3(Int3(dPos.x, dPos.y, z), stateSize)];
                    }

                    count += numMaps;
                }
            }

        activation += recurrentScalar * recurrentActivation;
    }
    
    int stateIndex = address3(pos, stateSize);

    states[stateIndex] = std::tanh(activation * std::sqrt(1.0f / count) * actScalar);
}

void LayerConv::create(ComputeSystem &cs, const Int3 &inputSize, int numMaps, int spatialFilterRadius, int spatialFilterStride, int recurrentFilterRadius) {
    this->inputSize = inputSize;
    this->numMaps = numMaps;

    spatial.filterRadius = spatialFilterRadius;
    this->spatialFilterStride = spatialFilterStride;

    spatial.filterDiam = spatial.filterRadius * 2 + 1;
    spatial.filterArea = spatial.filterDiam * spatial.filterDiam;

    states.resize(inputSize.x * inputSize.y * numMaps, 0.0f);

    if (recurrentFilterRadius < 0) {
        recurrent.filterRadius = -1;

        recurrent.filterDiam = 0;
        recurrent.filterArea = 0;

        statesPrev.clear();
    }
    else {
        recurrent.filterRadius = recurrentFilterRadius;

        recurrent.filterDiam = recurrent.filterRadius * 2 + 1;
        recurrent.filterArea = recurrent.filterDiam * recurrent.filterDiam;

        statesPrev = states;
    }

    paramsPerMap = spatial.filterArea * inputSize.z + recurrent.filterArea * numMaps + 1; // +1 for bias

    parameters.resize(paramsPerMap * numMaps, 0.0f);
}

void LayerConv::activate(ComputeSystem &cs, const FloatBuffer &inputStates) {
    Int3 stateSize = getStateSize();

    // Convolve
#ifdef KERNEL_NO_THREAD
    for (int x = 0; x < stateSize.x; x++)
        for (int y = 0; y < stateSize.y; y++)
            for (int z = 0; z < stateSize.z; z++)
                convolve(Int3(x, y, z), cs.rng, inputStates);
#else
    runKernel3(cs, std::bind(LayerConv::convolveKernel, std::placeholders::_1, std::placeholders::_2, this, inputStates), stateSize, cs.rng, cs.batchSize3);
#endif

    if (recurrent.filterRadius >= 0)
        statesPrev = states;
}