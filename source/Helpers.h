#pragma once

#include "ThreadPool.h"

#include <random>
#include <future>
#include <vector>
#include <array>
#include <functional>
#include <assert.h>

namespace swarm {
    class ComputeSystem;
    
    template <typename T> 
    struct Vec2 {
        T x, y;

        Vec2()
        {}

        Vec2(T X, T Y)
        : x(X), y(Y)
        {}
    };

    template <typename T> 
    struct Vec3 {
        T x, y, z;
        T pad;

        Vec3()
        {}

        Vec3(T X, T Y, T Z)
        : x(X), y(Y), z(Z)
        {}
    };

    template <typename T> 
    struct Vec4 {
        T x, y, z, w;

        Vec4()
        {}

        Vec4(T X, T Y, T Z, T W)
        : x(X), y(Y), z(Z), w(W)
        {}
    };

    typedef Vec2<int> Int2;
    typedef Vec3<int> Int3;
    typedef Vec4<int> Int4;
    typedef Vec2<float> Float2;
    typedef Vec3<float> Float3;
    typedef Vec4<float> Float4;

    typedef std::vector<int> IntBuffer;
    typedef std::vector<float> FloatBuffer;

    void runKernel1(ComputeSystem &cs, const std::function<void(int, std::mt19937 &rng)> &func, int size, std::mt19937 &rng, int batchSize);
    void runKernel2(ComputeSystem &cs, const std::function<void(const Int2 &, std::mt19937 &rng)> &func, const Int2 &size, std::mt19937 &rng, const Int2 &batchSize);
    void runKernel3(ComputeSystem &cs, const std::function<void(const Int3 &, std::mt19937 &rng)> &func, const Int3 &size, std::mt19937 &rng, const Int3 &batchSize);

    void fillInt(int pos, std::mt19937 &rng, IntBuffer &buffer, int fillValue);
    void fillFloat(int pos, std::mt19937 &rng, FloatBuffer &buffer, float fillValue);
    void copyInt(int pos, std::mt19937 &rng, const IntBuffer &src, IntBuffer &dst);
    void copyFloat(int pos, std::mt19937 &rng, const FloatBuffer &src, FloatBuffer &dst);

    inline bool inBounds0(const Int2 &position, const Int2 &upperBound) {
        return position.x >= 0 && position.x < upperBound.x && position.y >= 0 && position.y < upperBound.y;
    }

    inline bool inBounds(const Int2 &position, const Int2 &lowerBound, const Int2 &upperBound) {
        return position.x >= lowerBound.x && position.x < upperBound.x && position.y >= lowerBound.y && position.y < upperBound.y;
    }

    // Row-major ravels
    inline int address2(
        const Int2 &pos, // Position
        const Int2 &dims // Dimensions to ravel with
    ) {
        return pos.y + pos.x * dims.y;
    }

    inline int address3(
        const Int3 &pos, // Position
        const Int3 &dims // Dimensions to ravel with
    ) {
        return pos.z + pos.y * dims.z + pos.x * dims.z * dims.y;
    }

    inline int address4(
        const Int4 &pos, // Position
        const Int4 &dims // Dimensions to ravel with
    ) {
        return pos.w + pos.z * dims.w + pos.y * dims.w * dims.z + pos.x * dims.w * dims.z * dims.y;
    }

    float sigmoid(float x);

    float logit(float x);
}