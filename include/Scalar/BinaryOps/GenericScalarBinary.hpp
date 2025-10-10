/**
 * @file include/Scalar/BinaryOps/GenericScalarBinary.hpp
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

#define BINARY_SCALAR_OPERATOR(OPS, FUNC1, FUNC2, FUNC3)\
template <typename T1, typename T2, typename... Callables>                                            \
class Generic##OPS : public IVariable<Generic##OPS<T1, T2, Callables...>> {                           \
  private:                                                                                            \
    T1* mp_left{nullptr};                                                                             \
    T2* mp_right{nullptr};                                                                            \
    Tuples<Callables...> m_caller;                                                                    \
    DISABLE_COPY(Generic##OPS)                                                                        \
    DISABLE_MOVE(Generic##OPS)                                                                        \
  public:                                                                                             \
    const size_t m_nidx{};                                                                            \
    OMPair m_cache{};                                                                                 \
    constexpr Generic##OPS(T1* u, T2* v, Callables&&... call) : mp_left{u},                           \
                                                                mp_right{v},                          \
                                                                m_caller{std::make_tuple(std::forward<Callables>(call)...)},\
                                                                m_nidx{this->m_idx_count++}           \
    {}                                                                                                \
    V_OVERRIDE(Variable *symEval()) {                                                                 \
      ASSERT(false, "Invalid symbolic evaluation operation");                                         \
      return &Variable::t0;                                                                           \
    }                                                                                                 \
    V_OVERRIDE(Variable *symDeval(const Variable &var)) {                                             \
      ASSERT(false, "Invalid symbolic derivative operation");                                         \
      return &Variable::t0;                                                                           \
    }                                                                                                 \
    V_OVERRIDE(Type eval()) {                                                                         \
      const Type u = mp_left->eval();                                                                 \
      const Type v = mp_right->eval();                                                                \
      return FUNC1(u, v);                                                                             \
    }                                                                                                 \
    V_OVERRIDE(Type devalF(const Variable &var)) {                                                    \
      const Type du = mp_left->devalF(var);                                                           \
      const Type dv = mp_right->devalF(var);                                                          \
      const Type u = mp_left->eval();                                                                 \
      const Type v = mp_right->eval();                                                                \
      return FUNC2(u, v) * du + FUNC3(u, v) * dv;                                                     \
    }                                                                                                 \
    V_OVERRIDE(void traverse(OMPair* cache = nullptr)) {                                              \
      if (cache == nullptr) {                                                                         \
        cache = &m_cache;                                                                             \
        cache->reserve(g_map_reserve);                                                                \
        if (false == (*cache).empty()) {                                                              \
          (*cache).clear();                                                                           \
        }                                                                                             \
        if (false == mp_left->m_visited) {                                                            \
          mp_left->traverse(cache);                                                                   \
        }                                                                                             \
        if (false == mp_right->m_visited) {                                                           \
          mp_right->traverse(cache);                                                                  \
        }                                                                                             \
        const Type u = mp_left->eval();                                                               \
        const Type v = mp_right->eval();                                                              \
        (*cache)[mp_left->m_nidx] += (FUNC2(u, v));                                                   \
        (*cache)[mp_right->m_nidx] += (FUNC3(u, v));                                                  \
        std::for_each(EXECUTION_PAR mp_left->m_cache.begin(),                                         \
                      mp_left->m_cache.end(),                                                         \
                      [&cache, this, u, v](const auto &item) {                                        \
                        const auto idx = item.first;                                                  \
                        const auto val = item.second;                                                 \
                        (*cache)[idx] += (val * (FUNC2(u, v)));                                       \
                      });                                                                             \
        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),                                        \
                      mp_right->m_cache.end(),                                                        \
                      [&cache, this, u, v](const auto &item) {                                        \
                        const auto idx = item.first;                                                  \
                        const auto val = item.second;                                                 \
                        (*cache)[idx] += (val * (FUNC3(u, v)));                                       \
                      });                                                                             \
      } else {                                                                                        \
        const Type cCache = (*cache)[m_nidx];                                                         \
        if (false == mp_left->m_visited) {                                                            \
          mp_left->traverse(cache);                                                                   \
        }                                                                                             \
        if (false == mp_right->m_visited) {                                                           \
          mp_right->traverse(cache);                                                                  \
        }                                                                                             \
        const Type u = mp_left->eval();                                                               \
        const Type v = mp_right->eval();                                                              \
        (*cache)[mp_left->m_nidx] += ((FUNC2(u, v)) * cCache);                                        \
        (*cache)[mp_right->m_nidx] += ((FUNC3(u, v)) * cCache);                                       \
        if (cCache != (Type)(0)) {                                                                    \
          std::for_each(EXECUTION_PAR mp_left->m_cache.begin(),                                       \
                        mp_left->m_cache.end(),                                                       \
                        [&cache, this, cCache, u, v](const auto &item) {                              \
                          const auto idx = item.first;                                                \
                          const auto val = item.second;                                               \
                          (*cache)[idx] += (val * (FUNC2(u, v)) * cCache);                            \
                        });                                                                           \
          std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),                                      \
                        mp_right->m_cache.end(),                                                      \
                        [&cache, this, cCache, u, v](const auto &item) {                              \
                          const auto idx = item.first;                                                \
                          const auto val = item.second;                                               \
                          (*cache)[idx] += (val * (FUNC3(u, v)) * cCache);                            \
                        });                                                                           \
        }                                                                                             \
      }                                                                                               \
      if (false == mp_left->m_visited) {                                                              \
        mp_left->traverse(cache);                                                                     \
      }                                                                                               \
      if (false == mp_right->m_visited) {                                                             \
        mp_right->traverse(cache);                                                                    \
      }                                                                                               \
    }                                                                                                 \
    V_OVERRIDE(OMPair &getCache()) { return m_cache; }                                                \
    V_OVERRIDE(void reset()) { BINARY_RESET(); }                                                      \
    V_OVERRIDE(std::string_view getType() const) { return TOSTRING(Generic##OPS); }                   \
    bool findMe(void* v) const { BINARY_FIND_ME(); }                                                  \
    V_DTR(~Generic##OPS()) = default;                                                                 \
  };                                                                                                  \
template <typename T1, typename T2>                                                                   \
using CONCAT3(Generic, OPS, T1) = Generic##OPS<T1, T2, OpType>;                                       \
template <typename T1, typename T2>                                                                   \
constexpr const auto &OPS(const IVariable<T1> &u, const IVariable<T2> &v) {                           \
  auto tmp = Allocate <CONCAT3(Generic, OPS, T1)<T1,T2>>(const_cast<T1*>(static_cast<const T1*>(&u)), \
                                                         const_cast<T2*>(static_cast<const T2*>(&v)), \
                                                         OpObj);                                      \
  return *tmp;                                                                                        \
}                                                                                                     \
template <typename T, typename... Callables>                                                          \
class Generic##OPS<Type, T, Callables...> : public IVariable<Generic##OPS<Type, T, Callables...>> {   \
  private:                                                                                            \
    Type mp_left{0};                                                                                  \
    T* mp_right{nullptr};                                                                             \
    Tuples<Callables...> m_caller;                                                                    \
    DISABLE_COPY(Generic##OPS)                                                                        \
    DISABLE_MOVE(Generic##OPS)                                                                        \
  public:                                                                                             \
    const size_t m_nidx{};                                                                            \
    OMPair m_cache;                                                                                   \
    constexpr Generic##OPS(const Type& u, T* v, Callables&&... call) : mp_left{u},                    \
                                                                      mp_right{v},                   \
                                                                      m_caller{std::make_tuple(std::forward<Callables>(call)...)},\
                                                                      m_nidx{this->m_idx_count++}    \
    {}                                                                                               \
    V_OVERRIDE(Variable* symEval()) {                                                                \
      ASSERT(false, "Invalid symbolic evaluation operation");                                        \
      return &Variable::t0;                                                                          \
    }                                                                                                \
    V_OVERRIDE(Variable* symDeval(const Variable& var)) {                                            \
      ASSERT(false, "Invalid symbolic derivative operation");                                        \
      return &Variable::t0;                                                                          \
    }                                                                                                \
    V_OVERRIDE(Type eval()) {                                                                        \
      const Type v = mp_right->eval();                                                               \
      return FUNC1(mp_left, v);                                                                      \
    }                                                                                                \
    V_OVERRIDE(Type devalF(const Variable& var)) {                                                   \
      const Type dv = mp_right->devalF(var);                                                         \
      const Type v = mp_right->eval();                                                               \
      return FUNC3(mp_left, v) * dv;                                                                 \
    }                                                                                                \
    V_OVERRIDE(void traverse(OMPair* cache = nullptr)) {                                             \
      if (cache == nullptr) {                                                                        \
        cache = &m_cache;                                                                            \
        cache->reserve(g_map_reserve);                                                               \
        if (false == (*cache).empty()) {                                                             \
          (*cache).clear();                                                                          \
        }                                                                                            \
        if (false == mp_right->m_visited) {                                                          \
          mp_right->traverse(cache);                                                                 \
        }                                                                                            \
        const Type v = mp_right->eval();                                                             \
        (*cache)[mp_right->m_nidx] += (FUNC3(mp_left, v));                                           \
        std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),                                       \
                      mp_right->m_cache.end(),                                                       \
                      [&cache, this, v](const auto &item) {                                          \
                        const auto idx = item.first;                                                 \
                        const auto val = item.second;                                                \
                        (*cache)[idx] += (val * (FUNC3(mp_left, v)));                                \
                      });                                                                            \
      } else {                                                                                       \
        const Type cCache = (*cache)[m_nidx];                                                        \
        if (false == mp_right->m_visited) {                                                          \
          mp_right->traverse(cache);                                                                 \
        }                                                                                            \
        const Type v = mp_right->eval();                                                             \
        (*cache)[mp_right->m_nidx] += ((FUNC3(mp_left, v)) * cCache);                                \
        if (cCache != (Type)(0)) {                                                                   \
          std::for_each(EXECUTION_PAR mp_right->m_cache.begin(),                                     \
                        mp_right->m_cache.end(),                                                     \
                        [&cache, this, cCache, v](const auto &item) {                                \
                          const auto idx = item.first;                                               \
                          const auto val = item.second;                                              \
                          (*cache)[idx] += (val * (FUNC3(mp_left, v)) * cCache);                     \
                        });                                                                          \
        }                                                                                            \
      }                                                                                              \
      if (false == mp_right->m_visited) {                                                            \
        mp_right->traverse(cache);                                                                   \
      }                                                                                              \
    }                                                                                                \
    V_OVERRIDE(OMPair& getCache()) { return m_cache; }                                               \
    V_OVERRIDE(void reset()) { BINARY_RIGHT_RESET(); }                                               \
    V_OVERRIDE(std::string_view getType() const) { return TOSTRING(Generic##OPS); }                  \
    bool findMe(void* v) const { BINARY_RIGHT_FIND_ME(); }                                           \
    V_DTR(~Generic##OPS()) = default;                                                                \
};                                                                                                   \
template <typename T>                                                                                \
using CONCAT3(Generic, OPS, T2) = Generic##OPS<Type, T, OpType>;                                     \
template <typename T>                                                                                \
constexpr const auto& OPS(const Type& u, const IVariable<T>& v) {                                    \
  auto tmp = Allocate<CONCAT3(Generic, OPS, T2)<T>>(u, const_cast<T*>(static_cast<const T*>(&v)), OpObj);\
  return *tmp;                                                                                       \
}                                                                                                    \
template <typename T, typename... Callables>                                                         \
class Generic##OPS<T, Type, Callables...> : public IVariable<Generic##OPS<T, Type, Callables...>> {  \
private:                                                                                             \
  T* mp_left{nullptr};                                                                               \
  Type mp_right{0};                                                                                  \
  Tuples<Callables...> m_caller;                                                                     \
  DISABLE_COPY(Generic##OPS)                                                                         \
  DISABLE_MOVE(Generic##OPS)                                                                         \
public:                                                                                              \
  const size_t m_nidx{};                                                                             \
  OMPair m_cache;                                                                                    \
  constexpr Generic##OPS(T* u, const Type& v, Callables&&... call) : mp_left{u},                     \
                                                                     mp_right{v},                    \
                                                                     m_caller{std::make_tuple(std::forward<Callables>(call)...)},\
                                                                     m_nidx{this->m_idx_count++}     \
  {}                                                                                                 \
  V_OVERRIDE(Variable* symEval()) {                                                                  \
    ASSERT(false, "Invalid symbolic evaluation operation");                                          \
    return &Variable::t0;                                                                            \
  }                                                                                                  \
  V_OVERRIDE(Variable* symDeval(const Variable& var)) {                                              \
    ASSERT(false, "Invalid symbolic derivative operation");                                          \
    return &Variable::t0;                                                                            \
  }                                                                                                  \
  V_OVERRIDE(Type eval()) {                                                                          \
    const Type u = mp_left->eval();                                                                  \
    return FUNC1(u, mp_right);                                                                       \
  }                                                                                                  \
  V_OVERRIDE(Type devalF(const Variable &var)) {                                                     \
    const Type du = mp_left->devalF(var);                                                            \
    const Type u = mp_left->eval();                                                                  \
    return FUNC2(u, mp_right) * du;                                                                  \
  }                                                                                                  \
  V_OVERRIDE(void traverse(OMPair* cache = nullptr)) {                                               \
      if (cache == nullptr) {                                                                        \
        cache = &m_cache;                                                                            \
        cache->reserve(g_map_reserve);                                                               \
        if (false == (*cache).empty()) {                                                             \
          (*cache).clear();                                                                          \
        }                                                                                            \
        if (false == mp_left->m_visited) {                                                           \
          mp_left->traverse(cache);                                                                  \
        }                                                                                            \
        const Type u = mp_left->eval();                                                              \
        (*cache)[mp_left->m_nidx] += (FUNC2(u, mp_right));                                           \
        std::for_each(EXECUTION_PAR mp_left->m_cache.begin(),                                        \
                      mp_left->m_cache.end(),                                                        \
                      [&cache, this, u](const auto &item) {                                          \
                        const auto idx = item.first;                                                 \
                        const auto val = item.second;                                                \
                        (*cache)[idx] += (val * (FUNC2(u, mp_right)));                               \
                      });                                                                            \
      } else {                                                                                       \
        const Type cCache = (*cache)[m_nidx];                                                        \
        if (false == mp_left->m_visited) {                                                           \
          mp_left->traverse(cache);                                                                  \
        }                                                                                            \
        const Type u = mp_left->eval();                                                              \
        (*cache)[mp_left->m_nidx] += ((FUNC2(u, mp_right)) * cCache);                                \
        if (cCache != (Type)(0)) {                                                                   \
          std::for_each(EXECUTION_PAR mp_left->m_cache.begin(),                                      \
                        mp_left->m_cache.end(),                                                      \
                        [&cache, this, cCache, u](const auto &item) {                                \
                          const auto idx = item.first;                                               \
                          const auto val = item.second;                                              \
                          (*cache)[idx] += (val * (FUNC2(u, mp_right)) * cCache);                    \
                        });                                                                          \
        }                                                                                            \
      }                                                                                              \
      if (false == mp_left->m_visited) {                                                             \
        mp_left->traverse(cache);                                                                    \
      }                                                                                              \
    }                                                                                                \
    V_OVERRIDE(OMPair& getCache()) { return m_cache; }                                               \
    V_OVERRIDE(void reset()) { BINARY_LEFT_RESET(); }                                                \
    V_OVERRIDE(std::string_view getType() const) { return TOSTRING(Generic##OPS); }                  \
    bool findMe(void* v) const { BINARY_LEFT_FIND_ME(); }                                            \
    V_DTR(~Generic##OPS()) = default;                                                                \
};                                                                                                   \
template <typename T>                                                                                \
using CONCAT3(Generic, OPS, T3) = Generic##OPS<T, Type, OpType>;                                     \
template <typename T>                                                                                \
constexpr const auto& OPS(const IVariable<T>& v, const Type& u) {                                    \
  auto tmp = Allocate<CONCAT3(Generic, OPS, T3)<T>>(const_cast<T*>(static_cast<const T*>(&v)), u, OpObj);\
  return *tmp;                                                                                       \
}


