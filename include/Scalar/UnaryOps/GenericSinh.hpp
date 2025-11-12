/**
 * @file include/Scalar/UnaryOps/GenericSinh.hpp
 *
 * @copyright 2023-2025 Karthik Murali Madhavan Rathai
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
class GenericSinh : public IVariable<GenericSinh<T, Callables...>> {
private:
  // Resources
  T* mp_left{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericSinh)
    DISABLE_MOVE(GenericSinh)
  #endif

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD
  OMPair m_cache;

  // Constructor
  constexpr GenericSinh(T* u, Callables&&... call) :  mp_left{u}, 
                                                      m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                      m_nidx{this->m_idx_count++} 
  {}

  // Symbolic evaluation
  V_OVERRIDE(Variable* symEval()) {
    if (nullptr == this->mp_tmp) {
      auto tmp = Allocate<Expression>(sinh(EVAL_L()));
      this->mp_tmp = tmp.get();
    }
    return this->mp_tmp;
  }

  // Symbolic Differentiation
  V_OVERRIDE(Variable* symDeval(const Variable& var)) {
    // Static derivative computation
    if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end()) {
      auto tmp = Allocate<Expression>(cosh(EVAL_L()) * (DEVAL_L(var)));
      this->mp_dtmp[var.m_nidx] = tmp.get();
    }
    return this->mp_dtmp[var.m_nidx];
  }

  // Eval in run-time
  V_OVERRIDE(Type eval()) {
    // Returned evaluation
    const Type u = mp_left->eval();
    return (std::sinh(u));
  }

  // Deval in run-time for forward derivative
  V_OVERRIDE(Type devalF(const Variable& var)) {
    // Return derivative of sinh: (cosh(u))*ud
    const Type du = mp_left->devalF(var);
    const Type u = mp_left->eval();
    return (std::cosh(u) * du);
  }

  // Traverse run-time
  V_OVERRIDE(void traverse(OMPair* cache = nullptr)) {
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

      /* IMPORTANT: The derivative is computed here */
      const Type u = std::cosh(mp_left->eval());
      (*cache)[mp_left->m_nidx] += (u);

      // Modify cache for left node
      if (u != (Type)(0)) {
        std::for_each(EXECUTION_PAR mp_left->m_cache.begin(),
                      mp_left->m_cache.end(), [u, &cache](const auto &item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += (val * u);
                      });
      }
    } else {
      // Cached value
      const Type cCache = (*cache)[m_nidx];

      // Traverse left node
      if (false == mp_left->m_visited) {
        mp_left->traverse(cache);
      }

      /* IMPORTANT: The derivative is computed here */
      const Type ustar = (std::cosh(mp_left->eval()) * cCache);
      (*cache)[mp_left->m_nidx] += (ustar);

      // Modify cache for left node
      if (ustar != (Type)(0)) {
        std::for_each(EXECUTION_PAR mp_left->m_cache.begin(),
                      mp_left->m_cache.end(), [ustar, &cache](auto &item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += (val * ustar);
                      });
      }
    }
    // Traverse left/right nodes
    if (false == mp_left->m_visited) {
      mp_left->traverse(cache);
    }
  }

  // Get m_cache
  V_OVERRIDE(OMPair& getCache()) { 
    return m_cache; 
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    UNARY_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericSinh"; 
  }

  // Find me
  bool findMe(void* v) const { 
    UNARY_FIND_ME(); 
  }

  // Clone scalar expression
  constexpr const auto& cloneExp() const {
    return sinh(*mp_left);
  }

  // Destructor
  V_DTR(~GenericSinh()) = default;
};

// Generic sinh with 1 typename callables
template <typename T> 
using GenericSinhT = GenericSinh<T, OpType>;

// Function for sinh computation
template <typename T> 
constexpr const auto& sinh(const IVariable<T>& u) {
  const auto& _u = u.cloneExp();
  auto tmp = Allocate<GenericSinhT<T>>(const_cast<T*>(static_cast<const T*>(&_u)), OpObj);
  return *tmp;
}