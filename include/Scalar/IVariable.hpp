/**
 * @file include/IVariable.hpp
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
#include "MemoryManager.hpp"

// IVariable class to enforce expression templates for lazy evaluation
template <typename T>
class IVariable : public MetaVariable
{
protected:
    // Protected constructor
    IVariable() = default;
    // Protected destructor
    V_DTR(~IVariable()) = default;
};

// Binary left reset
#define BINARY_LEFT_RESET()                                                    \
    this->m_visited = false;                                                   \
    if (false == m_cache.empty())                                              \
    {                                                                          \
        m_cache.clear();                                                       \
    }                                                                          \
    MetaVariable::resetTemp();                                                 \
    mp_left->reset();


// Binary right reset
#define BINARY_RIGHT_RESET()                                                   \
    this->m_visited = false;                                                   \
    if (false == m_cache.empty())                                              \
    {                                                                          \
        m_cache.clear();                                                       \
    }                                                                          \
    MetaVariable::resetTemp();                                                 \
    mp_right->reset();

// Binary reset
#define BINARY_RESET()                                                         \
    this->m_visited = false;                                                   \
    if (false == m_cache.empty())                                              \
    {                                                                          \
        m_cache.clear();                                                       \
    }                                                                          \
    MetaVariable::resetTemp();                                                 \
    mp_left->reset();                                                          \
    mp_right->reset();

// Unary reset
#define UNARY_RESET()                                                          \
    this->m_visited = false;                                                   \
    if (false == m_cache.empty())                                              \
    {                                                                          \
        m_cache.clear();                                                       \
    }                                                                          \
    MetaVariable::resetTemp();                                                 \
    if (nullptr != mp_left)                                                    \
    {                                                                          \
        mp_left->reset();                                                      \
    }


#if defined(USE_CUSTOM_FUNCTIONS)
// Operations enum (Order matters!)
enum Op : size_t
{
    ADD = 0,
    SUB,
    MUL,
    DIV,
    SIN,
    COS,
    TAN,
    SINH,
    COSH,
    TANH,
    ASIN,
    ACOS,
    ATAN,
    ASINH,
    ACOSH,
    ATANH,
    ABS,
    SQRT,
    EXP,
    LOG,
    POW,
    RELU,
    COUNT
};

// Operation type (Order matters!)
#define OpType                                                                 \
    std::plus<Type>, std::minus<Type>, std::multiplies<Type>,                  \
        std::divides<Type>, Sin<Type>, Cos<Type>, Tan<Type>, Sinh<Type>,       \
        Cosh<Type>, Tanh<Type>, ASin<Type>, ACos<Type>, ATan<Type>,            \
        ASinh<Type>, ACosh<Type>, ATanh<Type>

// Operation objects (Order matters!)
#define OpObj                                                                  \
    std::plus<Type>(), std::minus<Type>(), std::multiplies<Type>(),            \
        std::divides<Type>(), Sin<Type>(), Cos<Type>(), Tan<Type>(),           \
        Sinh<Type>(), Cosh<Type>(), Tanh<Type>(), ASin<Type>(), ACos<Type>(),  \
        ATan<Type>(), ASinh<Type>(), ACosh<Type>(), ATanh<Type>()
#else
    struct X007{};
    #define OpType X007
    #define OpObj OpType()
#endif