/**
 * @file include/Scalar/BinaryOps/GenericBinaryC0Function.hpp
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
template <typename Func, typename FuncLHS, typename FuncRHS> 
constexpr const auto& BinaryC0Function(Func, FuncLHS, FuncRHS);

// Left/right side is an expression
template <typename Func, typename FuncLHS, typename FuncRHS>
class GenericBinaryC0Function : public IVariable<GenericBinaryC0Function<Func, FuncLHS, FuncRHS>> {
private:
  // Resources
  mutable Expression* mp_left{nullptr};
  mutable Expression* mp_right{nullptr};

  // Callables
  Func    m_f;
  FuncLHS m_flhs;
  FuncRHS m_frhs;

  template <typename T1, typename T2, typename = std::enable_if_t<is_valid_v<T1> && is_valid_v<T2>>> 
  constexpr const auto& setOperand(const T1& x1, const T2& x2) const {
    if constexpr (true == is_numeric_v<T1>) {
       mp_left = Allocate<Expression>(*Allocate<Parameter>(x1)).get();
    } else {
       mp_left = Allocate<Expression>(x1).get();
    }

    if constexpr (true == is_numeric_v<T2>) {
       mp_right = Allocate<Expression>(*Allocate<Parameter>(x2)).get();
    } else {
       mp_right = Allocate<Expression>(x2).get();
    }
    return *this;
  }  

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMPair m_cache{};

  // Constructor
  constexpr GenericBinaryC0Function(Func f, FuncLHS flhs, FuncRHS frhs) : m_f{f},
                                                                          m_flhs{flhs},
                                                                          m_frhs{frhs},
                                                                          m_nidx{this->m_idx_count++} 
  {}

  template <typename T1, typename T2, typename = std::enable_if_t<is_valid_v<T1> && is_valid_v<T2>>> 
  constexpr const auto& operator()(const T1& x1, const T2& x2) const {
    return BinaryC0Function(m_f, m_flhs, m_frhs).setOperand(x1,x2);
  }

  // Symbolic evaluation
  V_OVERRIDE(Variable *symEval()) {
    return &Variable::t0;
  }

  // Symbolic Differentiation
  V_OVERRIDE(Variable *symDeval(const Variable &var)) {
    return &Variable::t0;
  }

  // Eval in run-time
  V_OVERRIDE(Type eval()) {
    // Returned evaluation
    const Type u = Eval(*mp_left);
    const Type v = Eval(*mp_right);
    return m_f(u,v);
  }

  // Deval 1st in run-time for forward derivative
  V_OVERRIDE(Type devalF(const Variable &var)) {
    // Return derivative
    const Type du = DevalF(*mp_left, var);
    const Type dv = DevalF(*mp_right, var);
    // Returned evaluation
    const Type u = Eval(*mp_left);
    const Type v = Eval(*mp_right);
    // Chain rule
    return m_flhs(u,v)*du + m_frhs(u,v)*dv;
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
      const Type u = Eval(*mp_left);
      const Type v = Eval(*mp_right);
      (*cache)[mp_left->m_nidx] += m_flhs(u,v);
      (*cache)[mp_right->m_nidx] += m_frhs(u,v);

      // Modify cache for left node
      std::for_each(EXECUTION_PAR mp_left->m_cache.begin(),
                    mp_left->m_cache.end(), [&cache,this,u,v](const auto &item) {
                      const auto idx = item.first;
                      const auto val = item.second;
                      (*cache)[idx] += (val*m_flhs(u,v));
                    });

      // Modify cache for right node
      std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),
                    mp_right->m_cache.end(), [&cache,this,u,v](const auto &item) {
                      const auto idx = item.first;
                      const auto val = item.second;
                      (*cache)[idx] += (val*m_frhs(u,v));
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
      const Type u = Eval(*mp_left);
      const Type v = Eval(*mp_right);
      (*cache)[mp_left->m_nidx] += (m_flhs(u,v)*cCache);
      (*cache)[mp_right->m_nidx] += (m_frhs(u,v)*cCache);

      if (cCache != (Type)(0)) {
        // Modify cache for left node
        std::for_each(EXECUTION_PAR mp_left->m_cache.begin(),
                      mp_left->m_cache.end(),
                      [&cache,this,cCache,u,v](const auto &item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += (val * m_flhs(u,v)* cCache);
                      });

        // Modify cache for right node
        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),
                      mp_right->m_cache.end(),
                      [&cache,this,cCache,u,v](const auto &item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += (val * m_frhs(u,v) * cCache);
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
  V_OVERRIDE(OMPair &getCache()) { 
    return m_cache; 
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    BINARY_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericBinaryC0Function"; 
  }

  // Find me
  V_OVERRIDE(bool findMe(void *v) const) { 
    BINARY_FIND_ME(); 
  }

  // Destructor
  V_DTR(~GenericBinaryC0Function()) = default;
};

// Function computation
template <typename Func, typename FuncLHS, typename FuncRHS> 
constexpr const auto& BinaryC0Function(Func f, FuncLHS flhs, FuncRHS frhs) {
  static_assert(std::is_invocable_v<Func, Type, Type> == true, "Eval function is not invocable");
  static_assert(std::is_invocable_v<FuncLHS, Type, Type> == true, "Deval left function is not invocable");
  static_assert(std::is_invocable_v<FuncRHS, Type, Type> == true, "Deval right function is not invocable");
  auto tmp = Allocate<GenericBinaryC0Function<Func, FuncLHS, FuncRHS>>(f,flhs,frhs);
  return *tmp;
}
