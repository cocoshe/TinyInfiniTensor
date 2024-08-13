#include "core/allocator.h"
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        if (freeblock.empty()) {
            size_t offset = peak;
            used += size;
            peak += size;
            return offset;
        }

        for (auto it = freeblock.begin(); it != freeblock.end(); it ++) {
            size_t offset = it->first, blk_sz = it->second;
            if (size == blk_sz) {
                freeblock.erase(it);
                used += size;
                return offset;
            }
            if (size < blk_sz) {
                freeblock.erase(it);
                freeblock[offset + size] = blk_sz - size;
                used += size;
                return offset;
            }
        }
        // append to the last block
        auto it = freeblock.rbegin();
        size_t offset = it->first, blk_sz = it->second;
        freeblock.erase(offset);
        used += size;
        peak = offset + size;
        return offset;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        freeblock[addr] = size;
        used -= size;
        // merge freeblock
        for (auto it = freeblock.begin(); it != freeblock.end(); it ++ ) {
            size_t offset = it->first, blk_sz = it->second;
            if (freeblock.count(offset + blk_sz) > 0) {
                // merge
                size_t next_sz = freeblock[offset + blk_sz];
                freeblock[offset] += next_sz;
                freeblock.erase(offset + blk_sz);
                it --;
            }

        }

        // check the last block
        auto it = freeblock.rbegin();
        size_t offset = it->first, blk_sz = it->second;
        if (offset + blk_sz == peak) {
            used -= blk_sz;
            peak -= blk_sz;
            freeblock.erase(offset);
        }

    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
