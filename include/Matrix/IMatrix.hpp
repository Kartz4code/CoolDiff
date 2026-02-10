/**
 * @file include/Matrix/IMatrix.hpp
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

#include "MatOperators.hpp"
#include "MemoryManager.hpp"

// IMatrix class to enforce expression templates for lazy evaluation
template <typename T> 
class IMatrix : public MetaMatrix {
private:
  // CRTP const
  inline constexpr const T& derived() const {
    return static_cast<const T&>(*this);
  }

  // CRTP mutable
  inline constexpr T& derived() { 
    return static_cast<T&>(*this); 
  }

protected:
  // Protected constructor
  IMatrix() = default;

public:
  // Find me
  bool findMe(void* v) const {  
    return derived().findMe(v); 
  }

  // Clone matrix expression
  constexpr const auto& cloneExp() const {
    return derived().cloneExp();
  }
  
  // Memory strategy type
  constexpr std::string_view allocatorType() const {
    return derived().allocatorType();
  }

  // Protected destructor
  V_DTR(~IMatrix()) = default;
};

// Axis based operations
enum class Axis {
  ROW, COLUMN, ALL
};

// Matrix concatenation direction 
enum class ConcatAxis {
  HORIZONTAL, VERTICAL
};

// Binary matrix reset
#define BINARY_MAT_RESET()                                                     \
  this->m_visited = false;                                                     \
  this->clearClone();                                                          \
  if (false == m_cache.empty()) { m_cache.clear(); }                           \
  mp_left->reset();                                                            \
  mp_right->reset();                                                           \

#define BINARY_MAT_RIGHT_RESET()                                               \
  this->m_visited = false;                                                     \
  this->clearClone();                                                          \
  if (false == m_cache.empty()) { m_cache.clear(); }                           \
  mp_right->reset();                                                           \

// Matrix operations macros (shorthand)
#define MATRIX_ADD(...)              CoolDiff::TensorR2::MatOperators::MatrixAdd(__VA_ARGS__)
#define MATRIX_MUL(...)              CoolDiff::TensorR2::MatOperators::MatrixMul(__VA_ARGS__)
#define MATRIX_KRON(...)             CoolDiff::TensorR2::MatOperators::MatrixKron(__VA_ARGS__)
#define MATRIX_SUB(...)              CoolDiff::TensorR2::MatOperators::MatrixSub(__VA_ARGS__)
#define MATRIX_HADAMARD(...)         CoolDiff::TensorR2::MatOperators::MatrixHadamard(__VA_ARGS__)
#define MATRIX_SCALAR_ADD(...)       CoolDiff::TensorR2::MatOperators::MatrixScalarAdd(__VA_ARGS__)
#define MATRIX_SCALAR_MUL(...)       CoolDiff::TensorR2::MatOperators::MatrixScalarMul(__VA_ARGS__)
#define MATRIX_TRANSPOSE(...)        CoolDiff::TensorR2::MatOperators::MatrixTranspose(__VA_ARGS__)
#define MATRIX_DERV_TRANSPOSE(...)   CoolDiff::TensorR2::MatOperators::MatrixDervTranspose(__VA_ARGS__)
#define MATRIX_CONV(...)             CoolDiff::TensorR2::MatOperators::MatrixConv(__VA_ARGS__)
#define MATRIX_DERV_CONV(...)        CoolDiff::TensorR2::MatOperators::MatrixDervConv(__VA_ARGS__)
#define UNARY_OP_MAT(...)            CoolDiff::TensorR2::MatOperators::MatrixUnary(__VA_ARGS__)
#define MATRIX_INVERSE(...)          CoolDiff::TensorR2::MatOperators::MatrixInverse(__VA_ARGS__)
#define MATRIX_DET(...)              CoolDiff::TensorR2::MatOperators::MatrixDet(__VA_ARGS__)
#define MATRIX_TRACE(...)            CoolDiff::TensorR2::MatOperators::MatrixTrace(__VA_ARGS__)