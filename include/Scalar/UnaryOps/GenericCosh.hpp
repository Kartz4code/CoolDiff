/**
 * @file include/Scalar/UnaryOps/GenericCosh.hpp
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

template <typename T, typename... Callables>
class GenericCosh : public IVariable<GenericCosh<T, Callables...>>
{
private:
    // Resources
    T *mp_left{nullptr};

    // Callables
    Tuples<Callables...> m_caller;

    // Disable copy and move constructors/assignments
    DISABLE_COPY(GenericCosh)
    DISABLE_MOVE(GenericCosh)

public:
    // Block index
    const size_t m_nidx{};
    // Cache for reverse AD
    OMPair m_cache;

    // Constructor
    GenericCosh(T *u, Callables &&...call)
        : mp_left{u}, m_caller{std::make_tuple(
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

    // Symbolic evaluation
    V_OVERRIDE(Variable *symEval())
    {
        if (nullptr == this->mp_tmp)
        {
            auto tmp = Allocate<Expression>(cosh(EVAL_L()));
            this->mp_tmp = tmp.get();
        }
        return this->mp_tmp;
    }

    // Symbolic Differentiation
    V_OVERRIDE(Variable *symDeval(const Variable &var))
    {
        // Static derivative computation
        if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end())
        {
            auto tmp = Allocate<Expression>(sinh(EVAL_L()) * (DEVAL_L(var)));
            this->mp_dtmp[var.m_nidx] = tmp.get();
        }
        return this->mp_dtmp[var.m_nidx];
    }

    // Eval in run-time
    V_OVERRIDE(Type eval())
    {
        // Returned evaluation
        const Type u = mp_left->eval();
        return (std::cosh(u));
    }

    // Deval in run-time for forward derivative
    V_OVERRIDE(Type devalF(const Variable &var))
    {
        // Return derivative of cosh: (sinh(u))*ud
        const Type du = mp_left->devalF(var);
        const Type u = mp_left->eval();
        return (std::sinh(u) * du);
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

            /* IMPORTANT: The derivative is computed here */
            const Type u = std::sinh(mp_left->eval());
            (*cache)[mp_left->m_nidx] += (u);

            // Modify cache for left node
            if (u != 0)
            {
                for (const auto &[idx, val] : mp_left->m_cache)
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

            /* IMPORTANT: The derivative is computed here */
            const Type ustar = (std::sinh(mp_left->eval()) * cCache);
            (*cache)[mp_left->m_nidx] += (ustar);

            // Modify cache for left node
            if (ustar != 0)
            {
                for (const auto &[idx, val] : mp_left->m_cache)
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
    }

    // Get m_cache
    V_OVERRIDE(OMPair &getCache())
    {
        return m_cache;
    }

    // Reset visit run-time
    V_OVERRIDE(void reset())
    {
        UNARY_RESET();
    }

    // Get type
    V_OVERRIDE(std::string_view getType() const)
    {
        return "GenericCosh";
    }

    // Find me
    V_OVERRIDE(bool findMe(void *v) const)
    {
        UNARY_FIND_ME();
    }

    // Destructor
    V_DTR(~GenericCosh()) = default;
};

// Variable cosh with 1 typename callables
template <typename T>
using GenericCoshT = GenericCosh<T, OpType>;

// Function for cosh computation
template <typename T>
const GenericCoshT<T> &cosh(const IVariable<T> &u)
{
    auto tmp =
        Allocate<GenericCoshT<T>>(const_cast<T *>(static_cast<const T *>(&u)),
                                  OpObj);
    return *tmp;
}
