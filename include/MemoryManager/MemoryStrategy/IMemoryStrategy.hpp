/**
 * @file include/MemoryManager/include/MemoryManager/MemoryStrategy/IMemoryManager.hpp
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

#include "CommonHeader.hpp"

// IMemory strategy
template<typename T>
class IMemoryStrategy {
    public:
    IMemoryStrategy() = default;

    // Allocate 
    V_PURE(void allocate(size_t)); 

    // Deallocate 
    V_PURE(void deallocate() noexcept); 

    // Get raw pointer
    V_PURE(T* getPtr());

    // Get raw pointer const
    V_PURE(T* getPtr() const);

    // Set raw pointer 
    V_PURE(void setPtr(T*));

    // Reset pointer values
    V_PURE(void reset());

    // Get length of allocator memory 
    V_PURE(size_t getLength() const); 
     
    // Get allocator type
    V_PURE(std::string_view allocatorType() const);

    // Protected destructor
    V_DTR(~IMemoryStrategy()) = default;
};