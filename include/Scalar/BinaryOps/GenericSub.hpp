/**
 * @file include/Scalar/BinaryOps/GenericSub.hpp
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

// Left/right side is an expression
template <typename T1, typename T2, typename... Callables>
class GenericSub : public IVariable<GenericSub<T1, T2, Callables...>> {
private:
  // Resources
  T1 *mp_left{nullptr};
  T2 *mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericSub)
  DISABLE_MOVE(GenericSub)

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD
  OMPair m_cache;

  // Constructor
  constexpr GenericSub(T1 *u, T2 *v, Callables &&...call)
      : mp_left{u}, mp_right{v}, m_caller{std::make_tuple(
                                     std::forward<Callables>(call)...)},
        m_nidx{this->m_idx_count++} {}

  // Symbolic evaluation
  V_OVERRIDE(Variable *symEval()) {
    if (nullptr == this->mp_tmp) {
      auto tmp = Allocate<Expression>((EVAL_L()) - (EVAL_R()));
      this->mp_tmp = tmp.get();
    }
    return this->mp_tmp;
  }

  // Symbolic differentiation
  V_OVERRIDE(Variable *symDeval(const Variable &var)) {
    // Static derivative computation
    if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end()) {
      auto tmp = Allocate<Expression>((DEVAL_L(var)) - (DEVAL_R(var)));
      this->mp_dtmp[var.m_nidx] = tmp.get();
    }
    return this->mp_dtmp[var.m_nidx];
  }

  // Eval in run-time
  V_OVERRIDE(Type eval()) {
    // Returned evaluation
    const Type u = mp_left->eval();
    const Type v = mp_right->eval();
    return (u - v);
  }

  // Deval 1st in run-time for forward derivative
  V_OVERRIDE(Type devalF(const Variable &var)) {
    // Return derivative of -: ud - vd
    const Type du = mp_left->devalF(var);
    const Type dv = mp_right->devalF(var);
    return (du - dv);
  }

  // Traverse run-time
  V_OVERRIDE(void traverse(OMPair *cache = nullptr)) {
    // If cache is nullptr, i.e. for the first step
    if (cache == nullptr) {
      // cache is m_cache
      cache = &m_cache;
      cache->reserve(g_map_reserve);
      // Clear cache in the first entry
      if (false == (*cache).empty()) {
        (*cache).clear();
      }

      // Traverse left node
      if (false == mp_left->m_visited) {
        mp_left->traverse(cache);
      }
      // Traverse right node
      if (false == mp_right->m_visited) {
        mp_right->traverse(cache);
      }

      /* IMPORTANT: The derivative is computed here */
      (*cache)[mp_left->m_nidx] += (Type)(1);
      (*cache)[mp_right->m_nidx] += (Type)(-1);

      // Modify cache for left node
      std::for_each(EXECUTION_PAR mp_left->m_cache.begin(),
                    mp_left->m_cache.end(), [&cache](const auto &item) {
                      const auto idx = item.first;
                      const auto val = item.second;
                      (*cache)[idx] += (val);
                    });

      // Modify cache for right node
      std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),
                    mp_right->m_cache.end(), [&cache](const auto &item) {
                      const auto idx = item.first;
                      const auto val = item.second;
                      (*cache)[idx] += ((Type)(-1) * val);
                    });
    } else {
      // Cached value
      const Type cCache = (*cache)[m_nidx];

      // Traverse left node
      if (false == mp_left->m_visited) {
        mp_left->traverse(cache);
      }
      // Traverse right node
      if (false == mp_right->m_visited) {
        mp_right->traverse(cache);
      }

      /* IMPORTANT: The derivative is computed here */
      (*cache)[mp_left->m_nidx] += cCache;
      (*cache)[mp_right->m_nidx] += ((Type)(-1) * cCache);

      // Modify cache for left node
      if (cCache != (Type)(0)) {
        std::for_each(EXECUTION_PAR mp_left->m_cache.begin(),
                      mp_left->m_cache.end(),
                      [&cache, cCache](const auto &item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += (val * cCache);
                      });

        // Modify cache for right node
        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),
                      mp_right->m_cache.end(),
                      [&cache, cCache](const auto &item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += ((Type)(-1) * val * cCache);
                      });
      }
    }
    // Traverse left/right nodes
    if (false == mp_left->m_visited) {
      mp_left->traverse(cache);
    }
    if (false == mp_right->m_visited) {
      mp_right->traverse(cache);
    }
  }

  // Get m_cache
  V_OVERRIDE(OMPair &getCache()) { return m_cache; }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { BINARY_RESET(); }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { return "GenericSub"; }

  // Find me
  V_OVERRIDE(bool findMe(void *v) const) { BINARY_FIND_ME(); }

  // Destructor
  V_DTR(~GenericSub()) = default;
};

// GenericSub with 2 typename callables
template <typename T1, typename T2>
using GenericSubT1 = GenericSub<T1, T2, OpType>;

// Function for subtraction computation
template <typename T1, typename T2>
constexpr const auto &operator-(const IVariable<T1> &u,
                                const IVariable<T2> &v) {
  auto tmp = Allocate<GenericSubT1<T1, T2>>(
      const_cast<T1 *>(static_cast<const T1 *>(&u)),
      const_cast<T2 *>(static_cast<const T2 *>(&v)), OpObj);
  return *tmp;
}

// Left side is a number (subtraction)
template <typename T>
constexpr const auto &operator-(const Type &u, const IVariable<T> &v) {
  return (u + (Type)(-1) * (v));
}

// Right side is a number (subtraction)
template <typename T>
constexpr const auto &operator-(const IVariable<T> &v, const Type &u) {
  return (v + (Type)(-1) * (u));
}
