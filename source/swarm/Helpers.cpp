#include "Helpers.h"

#include "ComputeSystem.h"

using namespace swarm;

void swarm::runKernel1(
    ComputeSystem &cs,
    const std::function<void(int, std::mt19937 &)> &func,
    int size,
    std::mt19937 &rng,
    int batchSize
) {
    std::uniform_int_distribution<int> seedDist(0, 999999);

    // Ceil divide
    int batches = (size + batchSize - 1) / batchSize;

    #pragma omp parallel for
    for (int i = 0; i < batches; i++) {
        int itemBatchSize = std::min(size - i * batchSize, batchSize);
        
        std::mt19937 subRng(seedDist(rng));

        int pos = i * batchSize;

        for (int x = 0; x < itemBatchSize; x++)
            func(pos + x, subRng);
    }
}

void swarm::runKernel2(
    ComputeSystem &cs,
    const std::function<void(const Int2 &, std::mt19937 &)> &func,
    const Int2 &size, std::mt19937 &rng,
    const Int2 &batchSize
) {
    std::uniform_int_distribution<int> seedDist(0, 999999);

    // Ceil divide
    Int2 batches((size.x + batchSize.x - 1) / batchSize.x, (size.y + batchSize.y - 1) / batchSize.y);

    int totalBatches = batches.x * batches.y;

    #pragma omp parallel for
    for (int i = 0; i < totalBatches; i++) {
        int bx = i % batches.x;
        int by = (i / batches.x) % batches.y;

        Int2 itemBatchSize = Int2(std::min(size.x - bx * batchSize.x, batchSize.x), std::min(size.y - by * batchSize.y, batchSize.y));

        std::mt19937 subRng(seedDist(rng));
        Int2 pos(bx * batchSize.x, by * batchSize.y);

        for (int x = 0; x < itemBatchSize.x; x++)
            for (int y = 0; y < itemBatchSize.y; y++) {
                Int2 bPos;
                bPos.x = pos.x + x;
                bPos.y = pos.y + y;

                func(bPos, subRng);
            }
    }
}

void swarm::runKernel3(
    ComputeSystem &cs,
    const std::function<void(const Int3 &, std::mt19937 &)> &func,
    const Int3 &size,
    std::mt19937 &rng,
    const Int3 &batchSize
) {
    std::uniform_int_distribution<int> seedDist(0, 999999);

    // Ceil divide
    Int3 batches((size.x + batchSize.x - 1) / batchSize.x, (size.y + batchSize.y - 1) / batchSize.y, (size.z + batchSize.z - 1) / batchSize.z);

    int totalBatches = batches.x * batches.y * batches.z;
    
    #pragma omp parallel for
    for (int i = 0; i < totalBatches; i++) {
        int bx = i % batches.x;
        int by = (i / batches.x) % batches.y;
        int bz = (i / (batches.x * batches.y)) % batches.z;

        Int3 itemBatchSize = Int3(std::min(size.x - bx * batchSize.x, batchSize.x), std::min(size.y - by * batchSize.y, batchSize.y), std::min(size.z - bz * batchSize.z, batchSize.z));

        std::mt19937 subRng(seedDist(rng));
        Int3 pos(bx * batchSize.x, by * batchSize.y, bz * batchSize.z);

        for (int x = 0; x < itemBatchSize.x; x++)
            for (int y = 0; y < itemBatchSize.y; y++)
                for (int z = 0; z < itemBatchSize.z; z++) {
                    Int3 bPos;
                    bPos.x = pos.x + x;
                    bPos.y = pos.y + y;
                    bPos.z = pos.z + z;

                    func(bPos, subRng);
                }
    }
}

void swarm::fillInt(
    int pos,
    std::mt19937 &rng,
    IntBuffer* buffer,
    int fillValue
) {
    (*buffer)[pos] = fillValue;
}

void swarm::fillFloat(
    int pos,
    std::mt19937 &rng,
    FloatBuffer* buffer,
    float fillValue
) {
    (*buffer)[pos] = fillValue;
}

void swarm::copyInt(
    int pos,
    std::mt19937 &rng,
    const IntBuffer* src,
    IntBuffer* dst
) {
    (*dst)[pos] = (*src)[pos];
}

void swarm::copyFloat(
    int pos,
    std::mt19937 &rng,
    const FloatBuffer* src,
    FloatBuffer* dst
) {
    (*dst)[pos] = (*src)[pos];
}

std::vector<IntBuffer*> swarm::get(
    std::vector<std::shared_ptr<IntBuffer>> &v
) {
    std::vector<IntBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = v[i].get();

    return vp;
}

std::vector<FloatBuffer*> swarm::get(
    std::vector<std::shared_ptr<FloatBuffer>> &v
) {
    std::vector<FloatBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = v[i].get();

    return vp;
}

std::vector<const IntBuffer*> swarm::constGet(
    const std::vector<std::shared_ptr<IntBuffer>> &v
) {
    std::vector<const IntBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = v[i].get();

    return vp;
}

std::vector<const FloatBuffer*> swarm::constGet(
    const std::vector<std::shared_ptr<FloatBuffer>> &v
) {
    std::vector<const FloatBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = v[i].get();

    return vp;
}

std::vector<IntBuffer*> swarm::get(
    std::vector<IntBuffer> &v
) {
    std::vector<IntBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

std::vector<FloatBuffer*> swarm::get(
    std::vector<FloatBuffer> &v
) {
    std::vector<FloatBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

std::vector<const IntBuffer*> swarm::constGet(
    const std::vector<IntBuffer> &v
) {
    std::vector<const IntBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}

std::vector<const FloatBuffer*> swarm::constGet(
    const std::vector<FloatBuffer> &v
) {
    std::vector<const FloatBuffer*> vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = &v[i];

    return vp;
}