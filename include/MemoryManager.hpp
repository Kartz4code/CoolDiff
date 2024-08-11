/**
 * @file include/MemoryManager.hpp
 *
 * @copyright 2023-2024 Karthik Murali Madhavan Rathai
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

#include "MetaVariable.hpp"

class MemoryManager {
    private:
        // Allocate friend function
        template<typename T, typename... Args>
        friend SharedPtr<T> Allocate(Args&&... args);
        
        // Vector of deleted or to be deleted variables 
        inline static Vector<SharedPtr<MetaVariable>> m_del_ptr;

    public:
        // Get size of memory allocated
        static size_t size();
};

// Allocator
template<typename T, typename... Args>
SharedPtr<T> Allocate(Args&&... args) {
    const size_t size = sizeof(T);
    const size_t align = 0; 

    // Allocate with custom deleter
    SharedPtr<T> tmp{ new T(std::forward<Args>(args)...), DelPtr<T> };

    // Push the allocated object into stack to clear it later
    MemoryManager::m_del_ptr.push_back(tmp);
    return tmp;
}