/**
 * @file include/MemoryManager/include//MemoryManager/MemoryStrategy/GPUMemoryStrategy/GPUPinnedMemoryStrategy.hpp
 *
 * @copyright 2023-2026 Karthik Murali Madhavan Rathai
 */
/*
 * This file is part of CoolDiff library.
 *
 * You can redistribute it and/or modify it under the terms of the GNU
 * General Public License version 3 as published by the Free Software
 * Foundation.
 *
 * Licensees holding a valid commercial license may use this software
 * in accordance with the commercial license agreement provided in
 * conjunction with the software.  The terms and conditions of any such
 * commercial license agreement shall govern, supersede, and render
 * ineffective any application of the GPLv3 license to this software,
 * notwithstanding of any reference thereto in the software or
 * associated repository.
 */

#pragma once 

#include <cuda_runtime.h>
#include "IMemoryStrategy.hpp"

template<typename T>
class GPUPinnedMemoryStrategy : public IMemoryStrategy<T> {
    private:
        T* mp_ptr{nullptr}; 
        size_t m_length{}; 
        
    public:
        GPUPinnedMemoryStrategy() = default;

        // Allocate 
        V_OVERRIDE(void allocate(size_t N)) {
            ASSERT(N > 0, "Length is strictly non-negative");
            m_length = N; cudaMallocHost((void**)&mp_ptr, sizeof(T)*N); 
            ASSERT((nullptr != mp_ptr), "Local GPU allocation failed");
        }

        // Deallocate 
        V_OVERRIDE(void deallocate() noexcept) {
            if (nullptr != mp_ptr) {
                cudaFreeHost(mp_ptr); 
                mp_ptr = nullptr;
            }
        }

        // Get raw pointer
        V_OVERRIDE(T* getPtr()) {
            return mp_ptr;
        }

        // Get raw pointer const
        V_OVERRIDE(T* getPtr() const) {
            return mp_ptr;
        }

        // Set raw pointer 
        V_OVERRIDE(void setPtr(T* ptr)) {
           mp_ptr = ptr; 
        }

        // Reset pointer values
        V_OVERRIDE(void reset()) {
            if (nullptr != mp_ptr) {
                cudaMemset(mp_ptr, 0, sizeof(T)*m_length);
            }
        }

        // Get length of allocator memory 
        V_OVERRIDE(size_t getLength() const) {
            return m_length;
        }

        // Get allocator type
        V_OVERRIDE(std::string_view allocatorType() const) {
            return "GPUPinnedMemoryStrategy";
        }

        // Destructor
        V_DTR(~GPUPinnedMemoryStrategy()) = default;
};