/**
 * @file include/CommonHeader.hpp
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

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#ifndef BUILD_TYPE
#define SCALAR_TYPE double
#define USE_COMPLEX_MATH
#define USE_ROBIN_HOOD_MAP
#define USE_VIRTUAL_FUNCTIONS
#define USE_PARALLEL_POLICY
#endif

// Use parallel policy
#if defined(USE_PARALLEL_POLICY)
#include <execution>
#define EXECUTION_PAR std::execution::par,
#define EXECUTION_SEQ std::execution::seq,
constexpr const std::string_view g_execution_par = "Parallel";
constexpr const std::string_view g_execution_seq = "Sequential";
#endif

// Enable/disable copy/move operators
#ifndef ENABLE_COPY_MOVE
#define DISABLE_COPY(X)                                                        \
  X(const X &) = delete;                                                       \
  X &operator=(const X &) = delete;

#define DISABLE_MOVE(X)                                                        \
  X(X &&) noexcept = delete;                                                   \
  X &operator=(X &&) noexcept = delete;
#else
#define DISABLE_COPY(X)
#define DISABLE_MOVE(X)
#endif

// Eval/Deval left operator
#define EVAL_L() (*mp_left->symEval())
#define DEVAL_L(X) (*mp_left->symDeval(X))

// Eval/Deval right operator
#define EVAL_R() (*mp_right->symEval())
#define DEVAL_R(X) (*mp_right->symDeval(X))

// Unary find me
#define UNARY_FIND_ME()                                                        \
  if (static_cast<void *>(mp_left) == v) {                                     \
    return true;                                                               \
  } else if (mp_left->findMe(v) == true) {                                     \
    return true;                                                               \
  }                                                                            \
  return false;

// Binary find me
#define BINARY_FIND_ME()                                                       \
  if (static_cast<void *>(mp_left) == v) {                                     \
    return true;                                                               \
  } else if (static_cast<void *>(mp_right) == v) {                             \
    return true;                                                               \
  } else {                                                                     \
    if (mp_left->findMe(v) == true) {                                          \
      return true;                                                             \
    } else if (mp_right->findMe(v) == true) {                                  \
      return true;                                                             \
    }                                                                          \
  }                                                                            \
  return false;

// Binary right find me
#define BINARY_RIGHT_FIND_ME()                                                 \
  if (static_cast<void *>(mp_right) == v) {                                    \
    return true;                                                               \
  } else {                                                                     \
    if (mp_right->findMe(v) == true) {                                         \
      return true;                                                             \
    }                                                                          \
  }                                                                            \
  return false;

// Binary left find me
#define BINARY_LEFT_FIND_ME()                                                  \
  if (static_cast<void *>(mp_left) == v) {                                     \
    return true;                                                               \
  } else {                                                                     \
    if (mp_left->findMe(v) == true) {                                          \
      return true;                                                             \
    }                                                                          \
  }                                                                            \
  return false;

// Map reserve size
constexpr const inline size_t g_map_reserve{32};
// Constant size for vector of generic holder
constexpr const inline size_t g_vec_init{32};

// Typedef double as Type (TODO: Replace Type with a class/struct based on
// variants to support multiple types)
using Real = SCALAR_TYPE;

#if defined(USE_COMPLEX_MATH)
#include <complex>
using Type = std::complex<Real>;

Type operator+(Real, const Type &);
Type operator+(const Type &, Real);

Type operator-(Real, const Type &);
Type operator-(const Type &, Real);

Type operator*(Real, const Type &);
Type operator*(const Type &, Real);

Type operator/(Real, const Type &);
Type operator/(const Type &, Real);

bool operator!=(const Type &, Real);
bool operator!=(Real, const Type &);

bool operator==(const Type &, Real);
bool operator==(Real, const Type &);
#else
using Type = Real;
#endif

// Predeclare a few classes (Scalar)
class VarWrap;
class Variable;
class Parameter;
class Expression;

// Predeclare Matrix class
template <typename> class Matrix;

// Ordered map between size_t and Type
#if defined(USE_ROBIN_HOOD_MAP)
  #include <robin_hood.h>
  using OMPair = robin_hood::unordered_flat_map<size_t, Type>;
  using OMMatPair = robin_hood::unordered_flat_map<size_t, Matrix<Type>*>;
  // A generic unorderedmap
  template <typename T, typename U>
  using UnOrderedMap = robin_hood::unordered_flat_map<T, U>;
#else
  #include <unordered_map>
  using OMPair = std::unordered_map<size_t, Type>;
  using OMMatPair = std::unordered_map<size_t, Matrix<Type>*>;
  // A generic unorderedmap
  template <typename T, typename U> 
  using UnOrderedMap = std::unordered_map<T, U>;
#endif

// A generic vector type
template <typename T> using Vector = std::vector<T>;

// A generic variadic tuple type
template <typename... Args> using Tuples = std::tuple<Args...>;

// A generic shared pointer
template <typename T> using SharedPtr = std::shared_ptr<T>;

// A generic future
template <typename T> using Future = std::future<T>;

#if defined(USE_VIRTUAL_FUNCTIONS)
#define V_OVERRIDE(X) X override
#define V_DTR(X) virtual X
#define V_PURE(X) virtual X = 0
#endif

// Operations enum [Order matters!]
enum Op : size_t {
  ADD = 0,
  MUL,
  SUB,
  DIV,
  POW,
  SIN,
  COS,
  TAN,
  SINH,
  COSH,
  TANH,
  ASIN,
  ACOS,
  ATAN,
  ASINH,
  ACOSH,
  ATANH,
  SQRT,
  EXP,
  LOG,
  COUNT
};

// Enum classes
// Automatic differentiation modes
enum ADMode { FORWARD, REVERSE };

// Convert to string
template <typename T> std::string ToString(const T &value) {
  // If complex number
  if constexpr (std::is_same_v<T, std::complex<Real>>) {
    return std::move("(" + std::to_string(value.real()) + "," +
                     std::to_string(value.imag()) + ")");
  } else {
    return std::move(std::to_string(value));
  }
}