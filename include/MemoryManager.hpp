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