/**
 * @file include/Scalar/BinaryOps/GenericSum.hpp
 *
 * @copyright 2023-2026 Karthik Murali Madhavan Rathai
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
class GenericSum : public IVariable<GenericSum<T1, T2, Callables...>> {
private:
  // Resources
  T1* mp_left{nullptr};
  T2* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericSum)
    DISABLE_MOVE(GenericSum)
  #endif

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMPair m_cache{};

  // Constructor
  constexpr GenericSum(T1* u, T2* v, Callables&&... call) : mp_left{u}, 
                                                            mp_right{v}, 
                                                            m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                            m_nidx{this->m_idx_count++} 
  {}

  // Symbolic evaluation
  V_OVERRIDE(Variable* symEval()) {
    if (nullptr == this->mp_tmp) {
      auto tmp = Allocate<Expression>((EVAL_L()) + (EVAL_R()));
      this->mp_tmp = tmp.get();
    }
    return this->mp_tmp;
  }

  // Symbolic Differentiation
  V_OVERRIDE(Variable* symDeval(const Variable& var)) {
    // Static derivative computation
    if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end()) {
      auto tmp = Allocate<Expression>((DEVAL_L(var)) + (DEVAL_R(var)));
      this->mp_dtmp[var.m_nidx] = tmp.get();
    }
    return this->mp_dtmp[var.m_nidx];
  }

  // Eval in run-time
  V_OVERRIDE(Type eval()) {
    // Returned evaluation
    const Type u = mp_left->eval();
    const Type v = mp_right->eval();
    return (u + v);
  }

  // Deval 1st in run-time for forward derivative
  V_OVERRIDE(Type devalF(const Variable& var)) {
    // Return derivative of +: ud + vd
    const Type du = mp_left->devalF(var);
    const Type dv = mp_right->devalF(var);
    return (du + dv);
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
      // Traverse right node
      if (false == mp_right->m_visited) {
        mp_right->traverse(cache);
      }

      /* IMPORTANT: The derivative is computed here */
      (*cache)[mp_left->m_nidx] += (Type)(1);
      (*cache)[mp_right->m_nidx] += (Type)(1);

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
                      (*cache)[idx] += (val);
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
      (*cache)[mp_right->m_nidx] += cCache;

      if (cCache != (Type)(0)) {
        // Modify cache for left node
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
                        (*cache)[idx] += (val * cCache);
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
  V_OVERRIDE(OMPair& getCache()) { 
    return m_cache; 
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    BINARY_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericSum"; 
  }

  // Find me
  bool findMe(void* v) const { 
    BINARY_FIND_ME(); 
  }

  // Clone scalar expression
  constexpr const auto& cloneExp() const {
    return ((*mp_left) + (*mp_right));
  }

  // Destructor
  V_DTR(~GenericSum()) = default;
};

// Left/right side is a number
template <typename T, typename... Callables>
class GenericSum<Type, T, Callables...> : public IVariable<GenericSum<Type, T, Callables...>> {
private:
  // Resources
  Type mp_left{0};
  T* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  #if 0
    DISABLE_COPY(GenericSum)
    DISABLE_MOVE(GenericSum)
  #endif

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMPair m_cache;

  // Constructor
  constexpr GenericSum(const Type& u, T* v, Callables&&... call) :  mp_left{u}, 
                                                                    mp_right{v}, 
                                                                    m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                                    m_nidx{this->m_idx_count++} 
  {}

  // Symbolic evaluation
  V_OVERRIDE(Variable* symEval()) {
    // Evaluate variable in run-time
    if (nullptr == this->mp_tmp) {
      auto tmp = Allocate<Expression>((mp_left) + (EVAL_R()));
      this->mp_tmp = tmp.get();
    }
    return this->mp_tmp;
  }

  // Symbolic Differentiation
  V_OVERRIDE(Variable* symDeval(const Variable& var)) {
    // Static derivative computation
    if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end()) {
      auto tmp = Allocate<Expression>((DEVAL_R(var)));
      this->mp_dtmp[var.m_nidx] = tmp.get();
    }
    return this->mp_dtmp[var.m_nidx];
  }

  // Eval in run-time
  V_OVERRIDE(Type eval()) {
    // Returned evaluation
    const Type v = mp_right->eval();
    return (mp_left + v);
  }

  // Deval 1st in run-time for forward derivative
  V_OVERRIDE(Type devalF(const Variable& var)) {
    // Return derivative of sum : vd
    const Type dv = mp_right->devalF(var);
    return dv;
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

      // Traverse right node
      if (false == mp_right->m_visited) {
        mp_right->traverse(cache);
      }

      /* IMPORTANT: The derivative is computed here */
      (*cache)[mp_right->m_nidx] += (Type)1;

      // Modify cache for right node
      std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),
                    mp_right->m_cache.end(), [&cache](const auto &item) {
                      const auto idx = item.first;
                      const auto val = item.second;
                      (*cache)[idx] += (val);
                    });

    } else {
      // Cached value
      const Type cCache = (*cache)[m_nidx];

      // Traverse right node
      if (false == mp_right->m_visited) {
        mp_right->traverse(cache);
      }

      /* IMPORTANT: The derivative is computed here */
      (*cache)[mp_right->m_nidx] += cCache;

      // Modify cache for right node
      if (cCache != (Type)(0)) {
        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),
                      mp_right->m_cache.end(),
                      [&cache, cCache](const auto &item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += (val * cCache);
                      });
      }
    }

    // Traverse left/right nodes
    if (false == mp_right->m_visited) {
      if (nullptr != mp_right) {
        mp_right->traverse(cache);
      }
    }
  }

  // Get m_cache
  V_OVERRIDE(OMPair& getCache()) { 
    return m_cache; 
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    BINARY_RIGHT_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericSum"; 
  }

  // Find me
  bool findMe(void* v) const { 
    BINARY_RIGHT_FIND_ME(); 
  }

  // Clone scalar expression
  constexpr const auto& cloneExp() const {
    return ((mp_left) + (*mp_right));
  }

  // Destructor
  V_DTR(~GenericSum()) = default;
};

// GenericSum with 2 typename callables
template <typename T1, typename T2>
using GenericSumT1 = GenericSum<T1, T2, OpType>;

// GenericSum with 1 typename callables
template <typename T> 
using GenericSumT2 = GenericSum<Type, T, OpType>;

// Function for sum computation
template <typename T1, typename T2>
constexpr const auto& operator+(const IVariable<T1>& u, const IVariable<T2>& v) {
  const auto& _u = u.cloneExp();
  const auto& _v = v.cloneExp();
  auto tmp = Allocate<GenericSumT1<T1, T2>>(const_cast<T1*>(static_cast<const T1*>(&_u)),
                                            const_cast<T2*>(static_cast<const T2*>(&_v)), 
                                            OpObj);
  return *tmp;
}

// Left side is a number (sum)
template <typename T>
constexpr const auto& operator+(const Type& u, const IVariable<T>& v) {
  const auto& _v = v.cloneExp();
  auto tmp = Allocate<GenericSumT2<T>>(u, const_cast<T *>(static_cast<const T*>(&_v)), OpObj);
  return *tmp;
}

// Right side is a number (sum)
template <typename T>
constexpr const auto& operator+(const IVariable<T>& u, const Type& v) {
  const auto& _u = u.cloneExp();
  auto tmp = Allocate<GenericSumT2<T>>(v, const_cast<T*>(static_cast<const T*>(&_u)), OpObj);
  return *tmp;
}