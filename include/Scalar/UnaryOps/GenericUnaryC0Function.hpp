/**
 * @file include/Scalar/UnaryOps/GenericUnaryC1Function.hpp
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

// Predeclare Eval function
template <typename T> 
inline constexpr Type Eval(T&);
// Predeclare DevalF function
template <typename T>
inline constexpr Type DevalF(T&, const Variable&);

// Function computation
template <typename Func1, typename Func2> 
constexpr const auto& UnaryC0Function(Func1, Func2);

template <typename Func1, typename Func2>
class GenericUnaryC0Function : public IVariable<GenericUnaryC0Function<Func1, Func2>> {
private:
  // Resources
  mutable Expression* mp_left{nullptr};

  // Callables
  Func1 m_f1;
  Func2 m_f2;

  template <typename T, typename = std::enable_if_t<is_valid_v<T>>> 
  constexpr const auto& setOperand(const T& x) const {
    if constexpr (true == is_numeric_v<T>) {
       mp_left = Allocate<Expression>(*Allocate<Parameter>(x)).get();
    } else {
      mp_left = Allocate<Expression>(x).get();
    }
    return *this;
  }

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD
  OMPair m_cache;

  // Constructor
  constexpr GenericUnaryC0Function(Func1 f1, Func2 f2) : m_f1{f1},
                                                         m_f2{f2},
                                                         m_nidx{this->m_idx_count++} 
  {}

  template <typename T, typename = std::enable_if_t<is_valid_v<T>>> 
  constexpr const auto& operator()(const T& x) const {
    return UnaryC0Function(m_f1, m_f2).setOperand(x);
  }

  // Symbolic evaluation
  V_OVERRIDE(Variable *symEval()) {
    return &Variable::t0;
  }

  // Symbolic Differentiation
  V_OVERRIDE(Variable *symDeval(const Variable&)) {
    return &Variable::t0;
  }

  // Eval in run-time
  V_OVERRIDE(Type eval()) {
    // Returned evaluation
    const Type u = Eval(*mp_left);
    return (m_f1(u));
  }

  // Deval in run-time for forward derivative
  V_OVERRIDE(Type devalF(const Variable &var)) {
    // Return derivative
    const Type du = DevalF(*mp_left,var);
    const Type u = Eval(*mp_left);
    return (m_f2(u) * du);
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

      /* IMPORTANT: The derivative is computed here */
      const Type u = m_f2(Eval(*mp_left));
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
      const Type ustar = (m_f2(Eval(*mp_left)) * cCache);
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
  V_OVERRIDE(OMPair &getCache()) { 
    return m_cache; 
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    UNARY_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericUnaryC0Function"; 
  }

  // Find me
  V_OVERRIDE(bool findMe(void *v) const) { 
    UNARY_FIND_ME(); 
  }

  // Destructor
  V_DTR(~GenericUnaryC0Function()) = default;
};

// Function computation
template <typename Func1, typename Func2> 
constexpr const auto& UnaryC0Function(Func1 f1, Func2 f2) {
  static_assert(std::is_invocable_v<Func1, Type> == true, "Eval function is not invocable");
  static_assert(std::is_invocable_v<Func2, Type> == true, "Deval function is not invocable");
  auto tmp = Allocate<GenericUnaryC0Function<Func1, Func2>>(f1,f2);
  return *tmp;
}
