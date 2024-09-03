/**
 * @file include/Scalar/BinaryOps/GenericPow.hpp
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
class GenericPow : public IVariable<GenericPow<T1, T2, Callables...>> {
private:
  // Resources
  T1 *mp_left{nullptr};
  T2 *mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericPow)
  DISABLE_MOVE(GenericPow)

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD
  OMPair m_cache;

  // Constructor
  GenericPow(T1 *u, T2 *v, Callables &&...call)
      : mp_left{u}, mp_right{v}, m_caller{std::make_tuple(
                                     std::forward<Callables>(call)...)},
        m_nidx{this->m_idx_count++} {}


  // Symbolic evaluation
  V_OVERRIDE(Variable *symEval()) {
    if (nullptr == this->mp_tmp) {
      auto tmp = Allocate<Expression>(pow(EVAL_L(), EVAL_R()));
      this->mp_tmp = tmp.get();
    }
    return this->mp_tmp;
  }

  // Symbolic differentiation
  V_OVERRIDE(Variable *symDeval(const Variable &var)) {
    // Static derivative computation
    if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end()) {
      auto tmp = Allocate<Expression>(pow(EVAL_L(), EVAL_R()) *
                                      (((EVAL_R() / EVAL_L()) * DEVAL_L(var)) +
                                       (DEVAL_R(var) * log(EVAL_L()))));
      this->mp_dtmp[var.m_nidx] = tmp.get();
    }
    return this->mp_dtmp[var.m_nidx];
  }

  // Eval in run-time
  V_OVERRIDE(Type eval()) {
    // Returned evaluation
    const Type u = mp_left->eval();
    const Type v = mp_right->eval();
    return (Type)std::pow(u, v);
  }

  // Deval in run-time for forward derivative
  V_OVERRIDE(Type devalF(const Variable &var)) {
    // Return derivative of pow: pow(u,v)*(ud*v/u + vd*log(u))
    const Type du = mp_left->devalF(var);
    const Type dv = mp_right->devalF(var);
    const Type v = mp_right->eval();
    const Type u = mp_left->eval();
    return ((Type)std::pow(u, v) * (((v / u) * du) + (dv * log(u))));
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
      const Type v = mp_right->eval();
      const Type u = mp_left->eval();
      const Type pw = (Type)std::pow(u, v);
      const Type du = ((v / u) * pw);
      const Type dv = (log(u) * pw);
      (*cache)[mp_left->m_nidx] += du;
      (*cache)[mp_right->m_nidx] += dv;

      // Modify cache for left node
      if (du != (Type)(0)) {
        std::for_each(EXECUTION_PAR 
                      mp_left->m_cache.begin(), mp_left->m_cache.end(), 
                      [&cache,du](const auto& item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += (val*du);
                });
      }
      // Modify cache for right node
      if (dv != (Type)(0)) {
        std::for_each(EXECUTION_PAR 
                      mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                      [&cache,dv](const auto& item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += (val*dv);
                });
      }
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
      const Type v = mp_right->eval();
      const Type u = mp_left->eval();
      const Type pw = (Type)std::pow(u, v);
      const Type vstar = (((v / u) * pw) * cCache);
      const Type ustar = ((log(u) * pw) * cCache);
      (*cache)[mp_left->m_nidx] += (vstar);
      (*cache)[mp_right->m_nidx] += (ustar);

      // Modify cache for left node
      if (vstar != (Type)(0)) {
        std::for_each(EXECUTION_PAR 
                      mp_left->m_cache.begin(), mp_left->m_cache.end(), 
                      [&cache, vstar](const auto& item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += (val * vstar);
              });
      }
      // Modify cache for right node
      if (ustar != (Type)(0)) {
          std::for_each(EXECUTION_PAR 
                        mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                        [&cache, ustar](const auto& item) {
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
    if (false == mp_right->m_visited) {
      mp_right->traverse(cache);
    }
  }

  // Get m_cache
  V_OVERRIDE(OMPair &getCache()) { return m_cache; }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { BINARY_RESET(); }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { return "GenericPow"; }

  // Find me
  V_OVERRIDE(bool findMe(void *v) const) { BINARY_FIND_ME(); }

  // Destructor
  V_DTR(~GenericPow()) = default;
};

// Left/Right side is a number
template <typename T, typename... Callables>
class GenericPow<Type, T, Callables...>
    : public IVariable<GenericPow<Type, T, Callables...>> {
private:
  // Resources
  Type mp_left{0};
  T *mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericPow)
  DISABLE_MOVE(GenericPow)

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD
  OMPair m_cache;

  // Constructor
  GenericPow(const Type &u, T *v, Callables &&...call)
      : mp_left{u}, mp_right{v}, m_caller{std::make_tuple(
                                     std::forward<Callables>(call)...)},
        m_nidx{this->m_idx_count++} {}


  // Symbolic evaluation
  V_OVERRIDE(Variable *symEval()) {
    if (nullptr == this->mp_tmp) {
      auto tmp = Allocate<Expression>(pow(mp_left, EVAL_R()));
      this->mp_tmp = tmp.get();
    }
    return this->mp_tmp;
  }

  // Symbolic differentiation
  V_OVERRIDE(Variable *symDeval(const Variable &var)) {
    // Static derivative computation
    if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end()) {
      auto tmp = Allocate<Expression>(pow(mp_left, EVAL_R()) * DEVAL_R(var) *
                                      log(mp_left));
      this->mp_dtmp[var.m_nidx] = tmp.get();
    }
    return this->mp_dtmp[var.m_nidx];
  }

  // Eval in run-time
  V_OVERRIDE(Type eval()) {
    const Type v = mp_right->eval();
    return (Type)std::pow(mp_left, v);
  }

  // Deval in run-time for forward derivative
  V_OVERRIDE(Type devalF(const Variable &var)) {
    // Return derivative of pow: pow(u,v)*vd*log(u)
    const Type dv = mp_right->devalF(var);
    const Type v = mp_right->eval();
    return ((Type)std::pow(mp_left, v) * dv * log(mp_left));
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

      // Traverse right node
      if (false == mp_right->m_visited) {
        mp_right->traverse(cache);
      }

      /* IMPORTANT: The derivative is computed here */
      const Type v = mp_right->eval();
      const Type dv = ((Type)std::pow(mp_left, v) * log(mp_left));
      (*cache)[mp_right->m_nidx] += dv;

      // Modify cache for right node
      if (dv != (Type)(0)) {
      std::for_each(EXECUTION_PAR 
                    mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                    [&cache,dv](const auto& item) {
                      const auto idx = item.first;
                      const auto val = item.second;
                      (*cache)[idx] += (val*dv);
              });
      }
    } else {
      // Cached value
      const Type cCache = (*cache)[m_nidx];

      // Traverse right node
      if (false == mp_right->m_visited) {
        mp_right->traverse(cache);
      }

      /* IMPORTANT: The derivative is computed here */
      const Type v = mp_right->eval();
      const Type ustar = (cCache * ((Type)std::pow(mp_left, v) * log(mp_left)));
      (*cache)[mp_right->m_nidx] += (ustar);

      // Modify cache for right node
      if (ustar != (Type)(0)) {
        std::for_each(EXECUTION_PAR 
                      mp_right->m_cache.begin(), mp_right->m_cache.end(), 
                      [&cache, ustar](const auto& item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += (val * ustar);
              });
      }
    }
    // Traverse left/right nodes
    if (false == mp_right->m_visited) {
      mp_right->traverse(cache);
    }
  }

  // Get m_cache
  V_OVERRIDE(OMPair &getCache()) { return m_cache; }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { BINARY_RIGHT_RESET(); }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { return "GenericPow"; }

  // Find me
  V_OVERRIDE(bool findMe(void *v) const) { BINARY_RIGHT_FIND_ME(); }

  // Destructor
  V_DTR(~GenericPow()) = default;
};

// Right side is an expression
template <typename T, typename... Callables>
class GenericPow<T, Type, Callables...>
    : public IVariable<GenericPow<T, Type, Callables...>> {
private:
  // Resources
  T *mp_left{nullptr};
  Type mp_right{0};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericPow)
  DISABLE_MOVE(GenericPow)

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD
  OMPair m_cache;

  // Constructor
  GenericPow(T *u, const Type &v, Callables &&...call)
      : mp_left{u}, mp_right{v}, m_caller{std::make_tuple(
                                     std::forward<Callables>(call)...)},
        m_nidx{this->m_idx_count++} {}


  // Symbolic evaluation
  V_OVERRIDE(Variable *symEval()) {
    if (nullptr == this->mp_tmp) {
      auto tmp = Allocate<Expression>(pow(EVAL_L(), mp_right));
      this->mp_tmp = tmp.get();
    }
    return this->mp_tmp;
  }

  // Symbolic differentiation
  V_OVERRIDE(Variable *symDeval(const Variable &var)) {
    // Static derivative computation
    if (auto it = this->mp_dtmp.find(var.m_nidx); it == this->mp_dtmp.end()) {
      auto tmp = Allocate<Expression>(
          (mp_right * pow(EVAL_L(), mp_right - (Type)1) * DEVAL_L(var)));
      this->mp_dtmp[var.m_nidx] = tmp.get();
    }
    return this->mp_dtmp[var.m_nidx];
  }

  // Eval in run-time
  V_OVERRIDE(Type eval()) {
    // Returned evaluation
    const Type u = mp_left->eval();
    return ((Type)std::pow(u, mp_right));
  }

  // Deval in run-time for forward derivative
  V_OVERRIDE(Type devalF(const Variable &var)) {
    // Return derivative of pow: v*pow(u,v-1)*du
    const Type du = mp_left->devalF(var);
    const Type u = mp_left->eval();
    return (mp_right * (Type)std::pow(u, mp_right - (Type)1) * du);
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
      const Type u = mp_left->eval();
      const Type du = (mp_right * (Type)std::pow(u, mp_right - (Type)1));
      (*cache)[mp_left->m_nidx] += du;

      // Modify cache for left node
      if (du != (Type)(0)) {
        std::for_each(EXECUTION_PAR 
                      mp_left->m_cache.begin(), mp_left->m_cache.end(), 
                      [&cache,du](const auto& item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += (val*du);
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
      const Type u = mp_left->eval();
      const Type ustar =
          (mp_right * (Type)std::pow(u, mp_right - (Type)1)) * cCache;
      (*cache)[mp_left->m_nidx] += (ustar);

      // Modify cache for left node
      if (ustar != (Type)(0)) {
        std::for_each(EXECUTION_PAR 
                      mp_left->m_cache.begin(), mp_left->m_cache.end(), 
                      [&cache,ustar](const auto& item) {
                        const auto idx = item.first;
                        const auto val = item.second;
                        (*cache)[idx] += (val*ustar);
                });
      }
    }
    // Traverse left/right nodes
    if (false == mp_left->m_visited) {
      mp_left->traverse(cache);
    }
  }

  // Get m_cache
  V_OVERRIDE(OMPair &getCache()) { return m_cache; }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { BINARY_LEFT_RESET(); }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { return "GenericPow"; }

  // Find me
  V_OVERRIDE(bool findMe(void *v) const) { BINARY_LEFT_FIND_ME(); }

  // Virtual Destructor
  V_DTR(~GenericPow()) = default;
};

// GenericSum with 2 typename callables
template <typename T1, typename T2>
using GenericPowT1 = GenericPow<T1, T2, OpType>;

// Variable sum with 1 typename callables
template <typename T> using GenericPowT2 = GenericPow<Type, T, OpType>;

template <typename T> using GenericPowT3 = GenericPow<T, Type, OpType>;

// Function for power computation
template <typename T1, typename T2>
const GenericPowT1<T1, T2> &pow(const IVariable<T1> &u,
                                const IVariable<T2> &v) {
  auto tmp = Allocate<GenericPowT1<T1, T2>>(
      const_cast<T1 *>(static_cast<const T1 *>(&u)),
      const_cast<T2 *>(static_cast<const T2 *>(&v)), OpObj);
  return *tmp;
}

// Left side is a number (power)
template <typename T>
const GenericPowT2<T> &pow(const Type &u, const IVariable<T> &v) {
  auto tmp = Allocate<GenericPowT2<T>>(
      u, const_cast<T *>(static_cast<const T *>(&v)), OpObj);
  return *tmp;
}

// Right side is a number (power)
template <typename T>
const GenericPowT3<T> &pow(const IVariable<T> &u, const Type &v) {
  auto tmp = Allocate<GenericPowT3<T>>(
      const_cast<T *>(static_cast<const T *>(&u)), v, OpObj);
  return *tmp;
}
