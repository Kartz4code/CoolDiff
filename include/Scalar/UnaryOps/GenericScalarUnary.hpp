/**
 * @file include/Scalar/UnaryOps/GenericScalarUnary.hpp
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

#define UNARY_SCALAR_OPERATOR(OPS, FUNC1, FUNC2)\
template <typename T, typename... Callables>                                   \
class Generic##OPS : public IVariable<Generic##OPS<T, Callables...>> {         \
  private:                                                                     \
    T* mp_left{nullptr};                                                       \
    Tuples<Callables...> m_caller;                                             \
    DISABLE_COPY(Generic##OPS)                                                 \
    DISABLE_MOVE(Generic##OPS)                                                 \
  public:                                                                      \
    const size_t m_nidx{};                                                     \
    OMPair m_cache;                                                            \
    constexpr Generic##OPS(T* u, Callables&&... call)                          \
        : mp_left{u}, m_caller{std::make_tuple(                                \
                          std::forward<Callables>(call)...)},                  \
          m_nidx{this->m_idx_count++} {}                                       \
    V_OVERRIDE(Variable* symEval()) {                                          \
      ASSERT(false, "Invalid symbolic evaluation operation");                  \
      return &Variable::t0;                                                    \
    }                                                                          \
    V_OVERRIDE(Variable* symDeval(const Variable &)) {                         \
      ASSERT(false, "Invalid symbolic derivative operation");                  \
      return &Variable::t0;                                                    \
    }                                                                          \
    V_OVERRIDE(Type eval()) {                                                  \
      const Type u = mp_left->eval();                                          \
      return (FUNC1(u));                                                       \
    }                                                                          \
    V_OVERRIDE(Type devalF(const Variable& var)) {                             \
      const Type du = mp_left->devalF(var);                                    \
      const Type u = mp_left->eval();                                          \
      return (FUNC2(u) * du);                                                  \
    }                                                                          \
    V_OVERRIDE(void traverse(OMPair* cache = nullptr)) {                       \
    if (cache == nullptr) {                                                    \
        cache = &m_cache;                                                      \
        cache->reserve(g_map_reserve);                                         \
        if (false == (*cache).empty()) {                                       \
          (*cache).clear();                                                    \
        }                                                                      \
        if (false == mp_left->m_visited) {                                     \
          mp_left->traverse(cache);                                            \
        }                                                                      \
        const Type u = FUNC2(mp_left->eval());                                 \
        (*cache)[mp_left->m_nidx] += (u);                                      \
        if (u != (Type)(0)) {                                                  \
          std::for_each(EXECUTION_PAR mp_left->m_cache.begin(),                \
                        mp_left->m_cache.end(),                                \
                        [u, &cache](const auto &item) {                        \
                          const auto idx = item.first;                         \
                          const auto val = item.second;                        \
                        (*cache)[idx] += (val * u);                            \
                        });                                                    \
        }                                                                      \
      }                                                                        \
      else {                                                                   \
        const Type cCache = (*cache)[m_nidx];                                  \
        if (false == mp_left->m_visited) {                                     \
          mp_left->traverse(cache);                                            \
        }                                                                      \
        const Type ustar = (FUNC2(mp_left->eval()) * cCache);                  \
        (*cache)[mp_left->m_nidx] += (ustar);                                  \
        if (ustar != (Type)(0)) {                                              \
        std::for_each(EXECUTION_PAR mp_left->m_cache.begin(),                  \
                      mp_left->m_cache.end(), [ustar, &cache](auto &item) {    \
                        const auto idx = item.first;                           \
                        const auto val = item.second;                          \
                        (*cache)[idx] += (val * ustar);                        \
                      });                                                      \
        }                                                                      \
      }                                                                        \
      if (false == mp_left->m_visited) {                                       \
        mp_left->traverse(cache);                                              \
      }                                                                        \
    }                                                                          \
    V_OVERRIDE(OMPair& getCache()) { return m_cache; }                         \
    V_OVERRIDE(void reset()) { UNARY_RESET(); }                                \
    V_OVERRIDE(std::string_view getType() const) {                             \
      return TOSTRING(Generic##OPS);                                           \
    }                                                                          \
    bool findMe(void* v) const { UNARY_FIND_ME(); }                            \
    V_DTR(~Generic##OPS()) = default;                                          \
  };                                                                           \
template <typename T>                                                          \
using CONCAT3(Generic, OPS, T) = Generic##OPS<T, OpType>;                      \
template <typename T> constexpr const auto& OPS(const IVariable<T>& u) {       \
  auto tmp = Allocate < CONCAT3(Generic, OPS, T) < T >>                        \
               (const_cast<T *>(static_cast<const T *>(&u)), OpObj);           \
  return *tmp;                                                                 \
}

