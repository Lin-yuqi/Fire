#include "Fire/base/base.h"
#include <Fire/base/alloc.h>
#include <cuda_runtime_api.h>
#include <vector>

// ---------------base begin----------------
namespace base {

GPUAllocator::GPUAllocator() : DeviceAllocator(DeviceType::GPU) {}

void* GPUAllocator::allocate(size_t sz) const {
    int id = -1;

    cudaError_t state = cudaGetDevice(&id);
    CHECK(state == cudaSuccess);

    // ---------- 大于 1MB ----------
    if (sz > 1024 * 1024) {
        auto& big_buffers = _big_buffers_map[id];

        int sel_id = -1;

        for (int i = 0; i < big_buffers.size(); ++i) {
            if (big_buffers[i].byte_size >= sz &&
                !big_buffers[i].busy &&
                big_buffers[i].byte_size - sz < 1024 * 1024) {

                if (sel_id == -1 ||
                    big_buffers[i].byte_size < big_buffers[sel_id].byte_size) {
                    sel_id = i;
                }
            }
        }

        if (sel_id != -1) {
            big_buffers[sel_id].busy = true;
            return big_buffers[sel_id].data;
        }

        void* ptr = nullptr;

        state = cudaMalloc(&ptr, sz);
        if (state != cudaSuccess) {
            LOG(ERROR) << "cudaMalloc failed, size = " << sz;
            return nullptr;
        }

        big_buffers.emplace_back(ptr, sz, true);
        return ptr;
    }

    // ---------- 小于等于 1MB ----------
    auto& cuda_buffers = _cuda_buffers_map[id];

    for (int i = 0; i < cuda_buffers.size(); ++i) {
        if (cuda_buffers[i].byte_size >= sz &&
            !cuda_buffers[i].busy) {

            cuda_buffers[i].busy = true;

            _no_busy_cnt[id] -= cuda_buffers[i].byte_size;

            return cuda_buffers[i].data;
        }
    }

    void* ptr = nullptr;

    state = cudaMalloc(&ptr, sz);
    if (state != cudaSuccess) {
        LOG(ERROR) << "cudaMalloc failed, size = " << sz;
        return nullptr;
    }

    cuda_buffers.emplace_back(ptr, sz, true);

    return ptr;
}

void GPUAllocator::release(void* ptr) const {
    if (ptr == nullptr) {
        return;
    }

    cudaError_t state = cudaSuccess;

    // 1. 先回收过多的小 buffer cache
    for (auto& it : _cuda_buffers_map) {
        int device_id = it.first;

        if (_no_busy_cnt[device_id] > 1024ULL * 1024 * 1024) {

            state = cudaSetDevice(device_id);
            CHECK(state == cudaSuccess);

            auto& cuda_buffers = it.second;

            std::vector<CudaMemoryBuffer> remaining_buffers;

            for (auto& buffer : cuda_buffers) {
                if (!buffer.busy) {
                    state = cudaFree(buffer.data);
                    CHECK(state == cudaSuccess);
                } else {
                    remaining_buffers.push_back(buffer);
                }
            }

            cuda_buffers = std::move(remaining_buffers);
            _no_busy_cnt[device_id] = 0;
        }
    }

    // 2. 查找小 buffer
    for (auto& it : _cuda_buffers_map) {
        auto& cuda_buffers = it.second;

        for (auto& buffer : cuda_buffers) {
            if (buffer.data == ptr) {

                // 防止重复 release
                CHECK(buffer.busy);

                buffer.busy = false;
                _no_busy_cnt[it.first] += buffer.byte_size;

                return;
            }
        }
    }

    // 3. 查找大 buffer
    for (auto& it : _big_buffers_map) {
        auto& big_buffers = it.second;

        for (auto& buffer : big_buffers) {
            if (buffer.data == ptr) {

                CHECK(buffer.busy);

                buffer.busy = false;

                return;
            }
        }
    }

    // 4. 不属于 allocator cache 的指针，直接释放
    state = cudaFree(ptr);
    CHECK(state == cudaSuccess);
}

}  // namespace base