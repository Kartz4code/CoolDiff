#pragma once 

#include "MemoryManager.hpp"

// IVariable class to enforce expression templates for lazy evaluation
template <typename T>
class IVariable : public MetaVariable {
protected:  
    // Protected constructor
    IVariable() = default;
    // Protected destructor
    V_DTR( ~IVariable() ) = default;
};