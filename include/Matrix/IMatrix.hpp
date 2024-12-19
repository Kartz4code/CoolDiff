/**
 * @file include/Matrix/IMatrix.hpp
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

#include "MatOperators.hpp"
#include "MemoryManager.hpp"

// IVariable class to enforce expression templates for lazy evaluation
template <typename T> class IMatrix : public MetaMatrix {
private:
  // CRTP const
  inline constexpr const T &derived() const {
    return static_cast<const T &>(*this);
  }

  // CRTP mutable
  inline constexpr T &derived() { return static_cast<T &>(*this); }

protected:
  // Protected constructor
  IMatrix() = default;

public:
  // Find me
  bool findMe(void *v) const { return derived().findMe(v); }

  // Protected destructor
  V_DTR(~IMatrix()) = default;
};

// Binary matrix reset
#define BINARY_MAT_RESET()                                                     \
  this->m_visited = false;                                                     \
  mp_left->reset();                                                            \
  mp_right->reset();                                                           \
  std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [](Matrix<Type> *m) {   \
    if (nullptr != m) {                                                        \
      m->free();                                                               \
    }                                                                          \
  });

#define BINARY_MAT_RIGHT_RESET()                                               \
  this->m_visited = false;                                                     \
  mp_right->reset();                                                           \
  std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [](Matrix<Type> *m) {   \
    if (nullptr != m) {                                                        \
      m->free();                                                               \
    }                                                                          \
  });

template <typename T>
using ExpType = std::enable_if_t<true == std::is_base_of_v<MetaVariable, T> &&
                                 false == std::is_arithmetic_v<T> &&
                                 false == std::is_same_v<Type, T>>;

// Special matrices
enum MatrixSpl : size_t {
  ZEROS = 1 << 1,
  EYE = 1 << 2,
  ONES = 1 << 3,
  DIAG = 1 << 4,
  SYMM = 1 << 5
};

// Operations enum [Order matters!]
enum OpMat : size_t {
  ADD_MAT = 0,
  MUL_MAT,
  KRON_MAT,
  SUB_MAT,
  HADAMARD_MAT,
  ADD_SCALAR_MAT,
  MUL_SCALAR_MAT,
  TRANSPOSE_MAT,
  TRANSPOSE_DERV_MAT,
  CONV_MAT,
  CONV_DERV_MAT,
  SIN_MAT,
  COS_MAT,
  COUNT_MAT
};

// Matrix operations Macros
#define MATRIX_ADD(X, Y, Z) std::get<OpMat::ADD_MAT>(m_caller)(X, Y, Z)
#define MATRIX_MUL(X, Y, Z) std::get<OpMat::MUL_MAT>(m_caller)(X, Y, Z)
#define MATRIX_KRON(X, Y, Z) std::get<OpMat::KRON_MAT>(m_caller)(X, Y, Z)
#define MATRIX_SUB(X, Y, Z) std::get<OpMat::SUB_MAT>(m_caller)(X, Y, Z)
#define MATRIX_HADAMARD(X, Y, Z)                                               \
  std::get<OpMat::HADAMARD_MAT>(m_caller)(X, Y, Z)
#define MATRIX_SCALAR_ADD(X, Y, Z)                                             \
  std::get<OpMat::ADD_SCALAR_MAT>(m_caller)(X, Y, Z)
#define MATRIX_SCALAR_MUL(X, Y, Z)                                             \
  std::get<OpMat::MUL_SCALAR_MAT>(m_caller)(X, Y, Z)
#define MATRIX_TRANSPOSE(X, Y) std::get<OpMat::TRANSPOSE_MAT>(m_caller)(X, Y)
#define MATRIX_DERV_TRANSPOSE(X1, X2, X3, X4, X5, X6)                          \
  std::get<OpMat::TRANSPOSE_DERV_MAT>(m_caller)(X1, X2, X3, X4, X5, X6)
#define MATRIX_CONV(X1, X2, X3, X4, X5, X6, X7)                                \
  std::get<OpMat::CONV_MAT>(m_caller)(X1, X2, X3, X4, X5, X6, X7)
#define MATRIX_DERV_CONV(X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11)         \
  std::get<OpMat::CONV_DERV_MAT>(m_caller)(X1, X2, X3, X4, X5, X6, X7, X8, X9, \
                                           X10, X11)

#define MATRIX_SIN(X, Y) std::get<OpMat::SIN_MAT>(m_caller)(X, Y)
#define MATRIX_COS(X, Y) std::get<OpMat::COS_MAT>(m_caller)(X, Y)

// Operation type [Order matters!]
#define OpMatType                                                              \
  void (*)(const Matrix<Type> *, const Matrix<Type> *, Matrix<Type> *&),       \
      void (*)(const Matrix<Type> *, const Matrix<Type> *, Matrix<Type> *&),   \
      void (*)(const Matrix<Type> *, const Matrix<Type> *, Matrix<Type> *&),   \
      void (*)(const Matrix<Type> *, const Matrix<Type> *, Matrix<Type> *&),   \
      void (*)(const Matrix<Type> *, const Matrix<Type> *, Matrix<Type> *&),   \
      void (*)(Type, const Matrix<Type> *, Matrix<Type> *&),                   \
      void (*)(Type, const Matrix<Type> *, Matrix<Type> *&),                   \
      void (*)(const Matrix<Type> *, Matrix<Type> *&),                         \
      void (*)(const size_t, const size_t, const size_t, const size_t,         \
               const Matrix<Type> *, Matrix<Type> *&),                         \
      void (*)(const size_t, const size_t, const size_t, const size_t,         \
               const Matrix<Type> *, const Matrix<Type> *, Matrix<Type> *&),   \
      void (*)(const size_t, const size_t, const size_t, const size_t,         \
               const size_t, const size_t, const Matrix<Type> *,               \
               const Matrix<Type> *, const Matrix<Type> *,                     \
               const Matrix<Type> *, Matrix<Type> *&),                         \
      void (*)(const Matrix<Type> *, Matrix<Type> *&),                         \
      void (*)(const Matrix<Type> *, Matrix<Type> *&)

// Operation objects [Order matters!]
#define OpMatObj                                                               \
  MatrixAdd, MatrixMul, MatrixKron, MatrixSub, MatrixHadamard,                 \
      MatrixScalarAdd, MatrixScalarMul, MatrixTranspose, MatrixDervTranspose,  \
      MatrixConv, MatrixDervConv, MatrixSin, MatrixCos
