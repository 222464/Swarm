#include "Helpers.h"

#include "ComputeSystem.h"

using namespace swarm;

void swarm::runKernel1(ComputeSystem &cs, const std::function<void(int, std::mt19937 &)> &func, int size, std::mt19937 &rng, int batchSize) {
    std::uniform_int_distribution<int> seedDist(0, 999999);

    // Ceil divide
    int batches = (size + batchSize - 1) / batchSize;

    std::vector<std::future<void>> futures;

    // Create work items
    for (int x = 0; x < batches; x++) {
        int itemBatchSize = std::min(size - x * batchSize, batchSize);
        
        std::future<void> f = cs._pool.push([](int id, int seed, int pos, int batchSize, const std::function<void(int, std::mt19937 &)> &func) {
            std::mt19937 subRng(seed);

            for (int x = 0; x < batchSize; x++)
                func(pos + x, subRng);
        }, seedDist(rng), x * batchSize, itemBatchSize, func);

        futures.push_back(std::move(f));
    }

    // Wait
    for (int i = 0 ; i < futures.size(); i++)
        futures[i].wait();
}

void swarm::runKernel2(ComputeSystem &cs, const std::function<void(const Int2 &, std::mt19937 &)> &func, const Int2 &size, std::mt19937 &rng, const Int2 &batchSize) {
    std::uniform_int_distribution<int> seedDist(0, 999999);

    // Ceil divide
    Int2 batches((size.x + batchSize.x - 1) / batchSize.x, (size.y + batchSize.y - 1) / batchSize.y);

    std::vector<std::future<void>> futures;

    // Create work items
    for (int x = 0; x < batches.x; x++)
        for (int y = 0; y < batches.y; y++) {
            Int2 itemBatchSize = Int2(std::min(size.x - x * batchSize.x, batchSize.x), std::min(size.y - y * batchSize.y, batchSize.y));

            std::future<void> f = cs._pool.push([](int id, int seed, const Int2 &pos, const Int2 &batchSize, const std::function<void(const Int2 &, std::mt19937 &)> &func) {
                std::mt19937 subRng(seed);

                for (int x = 0; x < batchSize.x; x++)
                    for (int y = 0; y < batchSize.y; y++) {
                        Int2 bPos;
                        bPos.x = pos.x + x;
                        bPos.y = pos.y + y;

                        func(bPos, subRng);
                    }
            }, seedDist(rng), Int2(x * batchSize.x, y * batchSize.y), itemBatchSize, func);

            futures.push_back(std::move(f));
        }

    // Wait
    for (int i = 0 ; i < futures.size(); i++)
        futures[i].wait();
}

void swarm::runKernel3(ComputeSystem &cs, const std::function<void(const Int3 &, std::mt19937 &)> &func, const Int3 &size, std::mt19937 &rng, const Int3 &batchSize) {
    std::uniform_int_distribution<int> seedDist(0, 999999);

    // Ceil divide
    Int3 batches((size.x + batchSize.x - 1) / batchSize.x, (size.y + batchSize.y - 1) / batchSize.y, (size.z + batchSize.z - 1) / batchSize.z);

    std::vector<std::future<void>> futures;

    // Create work items
    for (int x = 0; x < batches.x; x++)
        for (int y = 0; y < batches.y; y++) 
            for (int z = 0; z < batches.z; z++) {
                Int3 itemBatchSize = Int3(std::min(size.x - x * batchSize.x, batchSize.x), std::min(size.y - y * batchSize.y, batchSize.y), std::min(size.z - z * batchSize.z, batchSize.z));

                std::future<void> f = cs._pool.push([](int id, int seed, const Int3 &pos, const Int3 &batchSize, const std::function<void(const Int3 &, std::mt19937 &)> &func) {
                    std::mt19937 subRng(seed);

                    for (int x = 0; x < batchSize.x; x++)
                        for (int y = 0; y < batchSize.y; y++)
                            for (int z = 0; z < batchSize.z; z++) {
                                Int3 bPos;
                                bPos.x = pos.x + x;
                                bPos.y = pos.y + y;
                                bPos.z = pos.z + z;

                                func(bPos, subRng);
                            }
                }, seedDist(rng), Int3(x * batchSize.x, y * batchSize.y, z * batchSize.z), itemBatchSize, func);

                futures.push_back(std::move(f));
            }

    // Wait
    for (int i = 0 ; i < futures.size(); i++)
        futures[i].wait();
}

void swarm::fillInt(int pos, std::mt19937 &rng, IntBuffer &buffer, int fillValue) {
    buffer[pos] = fillValue;
}

void swarm::fillFloat(int pos, std::mt19937 &rng, FloatBuffer &buffer, float fillValue) {
    buffer[pos] = fillValue;
}

void swarm::copyInt(int pos, std::mt19937 &rng, const IntBuffer &src, IntBuffer &dst) {
    dst[pos] = src[pos];
}

void swarm::copyFloat(int pos, std::mt19937 &rng, const FloatBuffer &src, FloatBuffer &dst) {
    dst[pos] = src[pos];
}

float swarm::sigmoid(float x) {
    if (x < 0.0f) {
        x = std::exp(x);

        return x / (1.0f + x);
    }

    return 1.0f / (1.0f + std::exp(-x));
}

float swarm::logit(float x) {
    return std::log(x / (1.0f - x));
}