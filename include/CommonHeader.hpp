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
#include <complex>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stddef.h>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

/* [Deprecated]
  Matrix transpose based matrix multiplication (Naive implementation) 
  #define MATRIX_TRANSPOSED_MUL
*/

// Use parallel policy
#if defined(USE_CXX_PARALLEL_POLICY)
  #include <execution>
  #define EXECUTION_SEQ std::execution::seq, 
  #define EXECUTION_PAR std::execution::par, 
#else 
  #define EXECUTION_SEQ
  #define EXECUTION_PAR 
#endif

// Eigen library
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#if defined(USE_COMPLEX_MATH)
  #if COOLDIFF_SCALAR_TYPE == 2
    using EigenMatrix = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  #elif COOLDIFF_SCALAR_TYPE == 1
    using EigenMatrix = Eigen::Matrix<std::complex<float>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  #else
    ASSERT(false, "Unknown type");
  #endif
#else
  #if COOLDIFF_SCALAR_TYPE == 2
    using EigenMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  #elif COOLDIFF_SCALAR_TYPE == 1
    using EigenMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  #else
    ASSERT(false, "Unknown type");
  #endif
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
constexpr inline static const size_t g_map_reserve{32};
// Constant size for vector of generic holder
constexpr inline static const size_t g_vec_init{32};

// Typedef double as Type (TODO: Replace Type with a class/struct based on
// variants to support multiple types)
#if COOLDIFF_SCALAR_TYPE == 1
  using Real = float;
#elif COOLDIFF_SCALAR_TYPE == 2
  using Real = double;
#else
  using Real = float;
#endif

#if defined(USE_COMPLEX_MATH)
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
template <typename> 
class Matrix;

// Predeclare Variable matrix
using MatVariable = Matrix<Variable>;
// Predeclare Parameter matrix
using MatParameter = Matrix<Parameter>;
// Predeclare Expression matrix
using MatExpression = Matrix<Expression>;
// Predeclare Type matrix
using MatType = Matrix<Type>;

// Pair type
template <typename T, typename U> 
using Pair = std::pair<T, U>;

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
  using OMMatPair = std::unordered_map<size_t, Matrix<Type> *>;
  // A generic unorderedmap
  template <typename T, typename U> 
  using UnOrderedMap = std::unordered_map<T, U>;
#endif

// A generic vector type
template <typename T> 
using Vector = std::vector<T>;

// A generic variadic tuple type
template <typename... Args> 
using Tuples = std::tuple<Args...>;

// A generic shared pointer
template <typename T> 
using SharedPtr = std::shared_ptr<T>;

// Function type
using FunctionType1 = std::function<Type(Type)>;

// A generic future
template <typename T> 
using Future = std::future<T>;

#if defined(USE_VIRTUAL_FUNCTIONS)
  #define V_OVERRIDE(X) X override
  #define V_UNPURE(X) virtual X
  #define V_DTR(X) virtual X
  #define V_PURE(X) virtual X = 0
#endif

// Enum classes - Automatic differentiation modes
enum ADMode { FORWARD, REVERSE };

// Convert to string
template <typename T> std::string ToString(const T &value) {
  // If complex number
  if constexpr (true == std::is_same_v<T, std::complex<Real>>) {
    return std::move( "(" + std::to_string(value.real()) + "," +
                      std::to_string(value.imag()) + ")"  );
  } else {
    return std::move(std::to_string(value));
  }
}

// Null pointer check
#if defined(USE_DYNAMIC_ASSERTIONS)
  // Check null poineter
  #define NULL_CHECK(PTR, MSG)                                                   \
    {                                                                            \
      if (nullptr == PTR) {                                                      \
        std::ostringstream oss;                                                  \
        oss << "[ERROR MSG]: " << MSG << "\n"                                    \
            << "[FILENAME]: " << __FILE__ << "\n"                                \
            << "[FUNCTION]: " << __FUNCTION__ << "\n"                            \
            << "[LINE NO]: " << __LINE__ << "\n";                                \
        std::cout << oss.str() << "\n";                                          \
        assert(false);                                                           \
      }                                                                          \
    }

  // Check boolean
  #define ASSERT(X, MSG)                                                         \
    {                                                                            \
      if (false == (X)) {                                                        \
        std::ostringstream oss;                                                  \
        oss << "[ERROR MSG]: " << MSG << "\n"                                    \
            << "[FILENAME]: " << __FILE__ << "\n"                                \
            << "[FUNCTION]: " << __FUNCTION__ << "\n"                            \
            << "[LINE NO]: " << __LINE__ << "\n";                                \
        std::cout << oss.str() << "\n";                                          \
        assert(false);                                                           \
      }                                                                          \
    }
#else
  #define NULL_CHECK(PTR, MSG)
  #define ASSERT(X, MSG)
#endif

// Time it base
#define TIME_IT(CODE, UNIT)                                                       \
  {                                                                               \
    auto start = std::chrono::high_resolution_clock::now();                       \
    CODE;                                                                         \
    auto stop = std::chrono::high_resolution_clock::now();                        \
    auto duration = std::chrono::duration_cast<std::chrono::UNIT>(stop - start);  \
    std::ostringstream oss;                                                       \
    oss << "[COMPUTATION TIME]: " << duration.count() << " " << #UNIT << "\n"     \
        << "[FILENAME]: " << __FILE__ << "\n"                                     \
        << "[FUNCTION]: " << __FUNCTION__ << "\n"                                 \
        << "[LINE NO]: " << __LINE__ << "\n";                                     \
    std::cout << oss.str() << "\n";                                               \
  }

// Time it in nanoseconds
#define TIME_IT_NS(CODE) TIME_IT(CODE, nanoseconds)
// Time it in microseconds
#define TIME_IT_US(CODE) TIME_IT(CODE, microseconds)
// Time it in milliseconds
#define TIME_IT_MS(CODE) TIME_IT(CODE, milliseconds)
// Time it in seconds
#define TIME_IT_S(CODE) TIME_IT(CODE, seconds)

// Range values from start to end
namespace CoolDiff{
  namespace Common {
      template <typename T> 
      class Range {
      private:
        Vector<T> m_vec;

      public:
        // Range constructor
        constexpr Range(T start, T end) : m_vec(end - start) {
          std::iota(m_vec.begin(), m_vec.end(), start);
        }

        constexpr typename Vector<T>::const_iterator begin() const {
          return m_vec.cbegin();
        }

        constexpr typename Vector<T>::const_iterator end() const {
          return m_vec.cend();
        }

        ~Range() = default;
      };
  }
}

// Hashing function for Pair<size_t, size_t>
template <> struct std::hash<Pair<size_t, size_t>> {
  std::size_t operator()(const Pair<size_t, size_t> &k) const {
    std::size_t res = 17;
    res = res * 31 + std::hash<size_t>()(k.first);
    res = res * 31 + std::hash<size_t>()(k.second);
    return res;
  }
};

// Common header that bears common attributes accross all classes
class CommonHeader {
  protected:
    // Index counter (A counter to count the number of matrix/scalar operations)
    inline static size_t m_idx_count{0};
};

// Useful macros for Generic unary/binary operators
#define STRINGIFY(X) #X
#define TOSTRING(X) STRINGIFY(X)
#define CONCAT3(STR1, STR2, STR3) STR1##STR2##STR3
