/**
 * @file include/BinaryOps/GenericProduct.hpp
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

#include "IVariable.hpp"
#include "Variable.hpp"

template <typename T1, typename T2, typename... Callables>
class GenericProduct : public IVariable<GenericProduct<T1, T2, Callables...>>
{
private:
    // Resources
    T1 *mp_left{nullptr};
    T2 *mp_right{nullptr};

    // Callables
    Tuples<Callables...> m_caller;

    // Disable copy and move constructors/assignments
    DISABLE_COPY(GenericProduct)
    DISABLE_MOVE(GenericProduct)

public:
    // Block index
    const size_t m_nidx{};
    // Cache for reverse AD 1st
    OMPair m_cache;

    // Constructor
    GenericProduct(T1 *u, T2 *v, Callables &&...call)
        : mp_left{u}, mp_right{v}, m_caller{std::make_tuple(
                                       std::forward<Callables>(call)...)},
          m_nidx{this->m_idx_count++}
    {
    }

    /*
    * ======================================================================================================
    * ======================================================================================================
    * ======================================================================================================
     _   _ ___________ _____ _   _  ___   _       _____  _   _ ___________ _     _____  ___ ______  _____
    | | | |_   _| ___ \_   _| | | |/ _ \ | |     |  _  || | | |  ___| ___ \ |   |  _  |/ _ \|  _  \/  ___|
    | | | | | | | |_/ / | | | | | / /_\ \| |     | | | || | | | |__ | |_/ / |   | | | / /_\ \ | | |\ `--.
    | | | | | | |    /  | | | | | |  _  || |     | | | || | | |  __||    /| |   | | | |  _  | | | | `--. \
    \ \_/ /_| |_| |\ \  | | | |_| | | | || |____ \ \_/ /\ \_/ / |___| |\ \| |___\ \_/ / | | | |/ / /\__/ /
     \___/ \___/\_| \_| \_/  \___/\_| |_/\_____/  \___/  \___/\____/\_| \_\_____/\___/\_| |_/___/  \____/

    *======================================================================================================
    *======================================================================================================
    *======================================================================================================
    */

    V_OVERRIDE(Variable *symEval())
    {
        // Evaluate variable in run-time
        if (nullptr == this->mp_tmp)
        {
            auto tmp = Allocate<Expression>((EVAL_R()) * (EVAL_L()));
            this->mp_tmp = tmp.get();
        }
        return this->mp_tmp;
    }

    // Differ
    V_OVERRIDE(Variable *symDeval(const Variable &var))
    {
        // Derivative of variable in run-time
        if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end())
        {
            auto tmp = Allocate<Expression>(((DEVAL_L(var)) * (EVAL_R())) +
                                            ((DEVAL_R(var)) * (EVAL_L())));
            this->mp_dtmp[var.m_nidx] = tmp.get();
        }
        return this->mp_dtmp[var.m_nidx];
    }

    // Eval in run-time
    V_OVERRIDE(Type eval())
    {
        // Returned evaluation
        const Type v = mp_right->eval();
        const Type u = mp_left->eval();
        return (u * v);
    }

    // Deval in run-time for forward derivative
    V_OVERRIDE(Type devalF(const Variable &var))
    {
        // Return derivative of *: ud*v + u*vd
        const Type du = mp_left->devalF(var);
        const Type dv = mp_right->devalF(var);
        const Type v = mp_right->eval();
        const Type u = mp_left->eval();
        return ((du * v) + (dv * u));
    }

    // Traverse run-time
    V_OVERRIDE(void traverse(OMPair *cache = nullptr))
    {
        // If cache is nullptr, i.e. for the first step
        if (cache == nullptr)
        {
            // cache is m_cache
            cache = &m_cache;
            cache->reserve(g_map_reserve);
            // Clear cache in the first entry
            if (false == (*cache).empty())
            {
                (*cache).clear();
            }

            // Traverse left node
            if (false == mp_left->m_visited)
            {
                mp_left->traverse(cache);
            }
            // Traverse right node
            if (false == mp_right->m_visited)
            {
                mp_right->traverse(cache);
            }

            /* IMPORTANT: The derivative is computed here */
            const Type v = mp_right->eval();
            const Type u = mp_left->eval();
            (*cache)[mp_left->m_nidx] += v;
            (*cache)[mp_right->m_nidx] += u;

            // Modify cache for left node
            if (v != 0)
            {
                for (const auto &[idx, val] : mp_left->m_cache)
                {
                    (*cache)[idx] += (val * v);
                }
            }
            // Modify cache for right node
            if (u != 0)
            {
                for (const auto &[idx, val] : mp_right->m_cache)
                {
                    (*cache)[idx] += (val * u);
                }
            }
        }
        else
        {
            // Cached value
            const Type cCache = (*cache)[m_nidx];

            // Traverse left node
            if (false == mp_left->m_visited)
            {
                mp_left->traverse(cache);
            }
            // Traverse right node
            if (false == mp_right->m_visited)
            {
                mp_right->traverse(cache);
            }

            /* IMPORTANT: The derivative is computed here */
            const Type vstar = (mp_right->eval() * cCache);
            const Type ustar = (mp_left->eval() * cCache);
            (*cache)[mp_left->m_nidx] += (vstar);
            (*cache)[mp_right->m_nidx] += (ustar);

            // Modify cache for left node
            if (vstar != 0)
            {
                for (const auto &[idx, val] : mp_left->m_cache)
                {
                    (*cache)[idx] += (val * vstar);
                }
            }
            // Modify cache for right node
            if (ustar != 0)
            {
                for (const auto &[idx, val] : mp_right->m_cache)
                {
                    (*cache)[idx] += (val * ustar);
                }
            }
        }

        // Traverse left/right nodes
        if (false == mp_left->m_visited)
        {
            mp_left->traverse(cache);
        }
        if (false == mp_right->m_visited)
        {
            mp_right->traverse(cache);
        }
    }

    // Get m_cache
    V_OVERRIDE(OMPair &getCache())
    {
        return m_cache;
    }

    // Reset visit run-time
    V_OVERRIDE(void reset())
    {
        BINARY_RESET();
    }

    // Get type
    V_OVERRIDE(std::string_view getType() const)
    {
        return "GenericProduct";
    }

    // Find me
    V_OVERRIDE(bool findMe(void *v) const)
    {
        BINARY_FIND_ME();
    }

    // Destructor
    V_DTR(~GenericProduct()) = default;
};

// Left/Right side is a number
template <typename T, typename... Callables>
class GenericProduct<Type, T, Callables...>
    : public IVariable<GenericProduct<Type, T, Callables...>>
{
private:
    // Resources
    Type mp_left{0};
    T *mp_right{nullptr};

    // Callables
    Tuples<Callables...> m_caller;

    // Disable copy and move constructors/assignments
    DISABLE_COPY(GenericProduct)
    DISABLE_MOVE(GenericProduct)

public:
    // Block index
    const size_t m_nidx{};
    // Cache for reverse AD 1st
    OMPair m_cache;

    // Constructor
    GenericProduct(const Type &u, T *v, Callables &&...call)
        : mp_left{u}, mp_right{v}, m_caller{std::make_tuple(
                                       std::forward<Callables>(call)...)},
          m_nidx{this->m_idx_count++}
    {
    }


    /*
    * ======================================================================================================
    * ======================================================================================================
    * ======================================================================================================
     _   _ ___________ _____ _   _  ___   _       _____  _   _ ___________ _     _____  ___ ______  _____
    | | | |_   _| ___ \_   _| | | |/ _ \ | |     |  _  || | | |  ___| ___ \ |   |  _  |/ _ \|  _  \/  ___|
    | | | | | | | |_/ / | | | | | / /_\ \| |     | | | || | | | |__ | |_/ / |   | | | / /_\ \ | | |\ `--.
    | | | | | | |    /  | | | | | |  _  || |     | | | || | | |  __||    /| |   | | | |  _  | | | | `--. \
    \ \_/ /_| |_| |\ \  | | | |_| | | | || |____ \ \_/ /\ \_/ / |___| |\ \| |___\ \_/ / | | | |/ / /\__/ /
     \___/ \___/\_| \_| \_/  \___/\_| |_/\_____/  \___/  \___/\____/\_| \_\_____/\___/\_| |_/___/  \____/

    *======================================================================================================
    *======================================================================================================
    *======================================================================================================
    */

    V_OVERRIDE(Variable *symEval())
    {
        // Evaluate variable in run-time
        if (nullptr == this->mp_tmp)
        {
            auto tmp = Allocate<Expression>((mp_left) * (EVAL_R()));
            this->mp_tmp = tmp.get();
        }
        return this->mp_tmp;
    }

    // Differ
    V_OVERRIDE(Variable *symDeval(const Variable &var))
    {
        // Derivative of variable in run-time
        if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end())
        {
            auto tmp = Allocate<Expression>((mp_left) * (DEVAL_R(var)));
            this->mp_dtmp[var.m_nidx] = tmp.get();
        }
        return this->mp_dtmp[var.m_nidx];
    }

    // Eval in run-time
    V_OVERRIDE(Type eval())
    {
        const Type v = mp_right->eval();
        return (mp_left * v);
    }

    // Deval in run-time for forward derivative
    V_OVERRIDE(Type devalF(const Variable &var))
    {
        // Return derivative of *: mp_left*vd
        const Type dv = mp_right->devalF(var);
        return (mp_left * dv);
    }

    // Traverse run-time
    V_OVERRIDE(void traverse(OMPair *cache = nullptr))
    {
        // If cache is nullptr, i.e. for the first step
        if (cache == nullptr)
        {
            // cache is m_cache
            cache = &m_cache;
            cache->reserve(g_map_reserve);
            // Clear cache in the first entry
            if (false == (*cache).empty())
            {
                (*cache).clear();
            }

            // Traverse right node
            if (false == mp_right->m_visited)
            {
                mp_right->traverse(cache);
            }

            /* IMPORTANT: The derivative is computed here */
            (*cache)[mp_right->m_nidx] += mp_left;

            // Modify cache for right node
            if (mp_left != 0)
            {
                for (const auto &[idx, val] : mp_right->m_cache)
                {
                    (*cache)[idx] += (val * mp_left);
                }
            }
        }
        else
        {
            // Cached value
            const Type cCache = (*cache)[m_nidx];

            // Traverse right node
            if (false == mp_right->m_visited)
            {
                mp_right->traverse(cache);
            }

            /* IMPORTANT: The derivative is computed here */
            const Type ustar = (cCache * mp_left);
            (*cache)[mp_right->m_nidx] += (ustar);

            // Modify cache for right node
            if (ustar != 0)
            {
                for (const auto &[idx, val] : mp_right->m_cache)
                {
                    (*cache)[idx] += (val * ustar);
                }
            }
        }
        // Traverse left/right nodes
        if (false == mp_right->m_visited)
        {
            mp_right->traverse(cache);
        }
    }

    // Get m_cache
    V_OVERRIDE(OMPair &getCache())
    {
        return m_cache;
    }

    // Reset visit run-time
    V_OVERRIDE(void reset())
    {
        BINARY_RIGHT_RESET();
    }

    // Get type
    V_OVERRIDE(std::string_view getType() const)
    {
        return "GenericProduct";
    }

    // Find me
    V_OVERRIDE(bool findMe(void *v) const)
    {
        BINARY_RIGHT_FIND_ME();
    }

    // Destructor
    V_DTR(~GenericProduct()) = default;
};

// GenericProduct with 2 typename callables
template <typename T1, typename T2>
using GenericProductT1 = GenericProduct<T1, T2, OpType>;

// GenericProduct with 1 typename callables
template <typename T>
using GenericProductT2 = GenericProduct<Type, T, OpType>;

// Function for product computation
template <typename T1, typename T2>
const GenericProductT1<T1, T2> &operator*(const IVariable<T1> &u,
                                          const IVariable<T2> &v)
{

    auto tmp = Allocate<GenericProductT1<T1, T2>>(
        const_cast<T1 *>(static_cast<const T1 *>(&u)),
        const_cast<T2 *>(static_cast<const T2 *>(&v)),
        OpObj);
    return *tmp;
}

// Left side is a number (product)
template <typename T>
const GenericProductT2<T> &operator*(const Type &u, const IVariable<T> &v)
{
    auto tmp = Allocate<GenericProductT2<T>>(
        u,
        const_cast<T *>(static_cast<const T *>(&v)),
        OpObj);
    return *tmp;
}

// Right side is a number (product)
template <typename T>
const GenericProductT2<T> &operator*(const IVariable<T> &u, const Type &v)
{
    auto tmp = Allocate<GenericProductT2<T>>(
        v,
        const_cast<T *>(static_cast<const T *>(&u)),
        OpObj);
    return *tmp;
}

// Right side is a number (division)
template <typename T>
const GenericProductT2<T> &operator/(const IVariable<T> &u, const Type &v)
{
    auto tmp = Allocate<GenericProductT2<T>>(
        (1 / v),
        const_cast<T *>(static_cast<const T *>(&u)),
        OpObj);
    return *tmp;
}
