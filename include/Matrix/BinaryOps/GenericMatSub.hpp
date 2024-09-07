/**
 * @file include/Matrix/BinaryOps/GenericMatSub.hpp
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

#include "IMatrix.hpp"
#include "Matrix.hpp"

// Left/right side is an expression
template <typename T1, typename T2, typename... Callables>
class GenericMatSub : public IMatrix<GenericMatSub<T1, T2, Callables...>> {
private:
  // Resources
  T1 *mp_left{nullptr};
  T2 *mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatSub)
  DISABLE_MOVE(GenericMatSub)

  // Verify dimensions of result matrix for subtraction operation
  inline constexpr bool verifyDim() const {
    // Left matrix rows
    const int lr = mp_left->getNumRows();
    // Right matrix rows
    const int rr = mp_right->getNumRows();
    // Left matrix columns
    const int lc = mp_left->getNumColumns();
    // Right matrix columns
    const int rc = mp_right->getNumColumns();
    // Condition for Matrix-Matrix subtraction
    return ((lr == rr) && (lc == rc));
  }

public:
  // Result
  Matrix<Type> *mp_result{nullptr};
  // Derivative result
  Matrix<Type> *mp_dresult{nullptr};

  // Block index
  const size_t m_nidx{};

  // Constructor
  GenericMatSub(T1 *u, T2 *v, Callables &&...call)
      : mp_left{u}, mp_right{v}, mp_result{nullptr}, mp_dresult{nullptr},
        m_caller{std::make_tuple(std::forward<Callables>(call)...)},
        m_nidx{this->m_idx_count++} {}

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { return mp_left->getNumRows(); }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { return mp_right->getNumColumns(); }

  // Find me
  bool findMe(void *v) const { BINARY_FIND_ME(); }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type> *eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix subtraction dimensions mismatch");

    // Get raw pointers to result, left and right matrices
    Matrix<Type> *left_mat = mp_left->eval();
    Matrix<Type> *right_mat = mp_right->eval();

    // Matrix-Matrix subtraction computation (Policy design)
    MATRIX_SUB(left_mat, right_mat, mp_result);

    // Return result pointer
    return mp_result;
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix subtraction dimensions mismatch");

    // Left and right matrices
    Matrix<Type> *dleft_mat = mp_left->devalF(X);
    Matrix<Type> *dright_mat = mp_right->devalF(X);

    // Matrix-Matrix derivative subtraction computation (Policy design)
    MATRIX_SUB(dleft_mat, dright_mat, mp_dresult);

    // Return result pointer
    return mp_dresult;
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()){BINARY_MAT_RESET()}

  // Get type
  V_OVERRIDE(std::string_view getType() const) {
    return "GenericMatSub";
  }

  // Destructor
  V_DTR(~GenericMatSub()) = default;
};

// GenericMatSub with 2 typename callables
template <typename T1, typename T2>
using GenericMatSubT = GenericMatSub<T1, T2, OpMatType>;

// Function for sub computation
template <typename T1, typename T2>
const GenericMatSubT<T1, T2> &operator-(const IMatrix<T1> &u,
                                        const IMatrix<T2> &v) {
  auto tmp = Allocate<GenericMatSubT<T1, T2>>(
      const_cast<T1 *>(static_cast<const T1 *>(&u)),
      const_cast<T2 *>(static_cast<const T2 *>(&v)), OpMatObj);
  return *tmp;
}

// Matrix sub with scalar (LHS) - SFINAE'd
template<typename T, typename Z, typename = std::enable_if_t<std::is_base_of_v<MetaVariable, Z> && 
                                                             !std::is_arithmetic_v<Z> &&
                                                             !std::is_same_v<Type,Z>>>
const auto& operator-(const Z& value, const IMatrix<T>& mat) {
  // Create type matrix filled with value (Type)
  auto& u = CreateMatrix<Expression>(mat.getNumRows(), mat.getNumColumns()); 
  std::fill_n(EXECUTION_PAR u.getMatrixPtr(), u.getNumElem(), value);
  // Return matrix
  return u - mat;
}

// Matrix sub with scalar (RHS) - SFINAE'd
template<typename T, typename Z, typename = std::enable_if_t<std::is_base_of_v<MetaVariable, Z> && 
                                                             !std::is_arithmetic_v<Z> &&
                                                             !std::is_same_v<Type,Z>>>
const auto& operator-(const IMatrix<T>& mat, const Z& value) {
  // Create type matrix filled with value (Type)
  auto& u = CreateMatrix<Expression>(mat.getNumRows(), mat.getNumColumns()); 
  std::fill_n(EXECUTION_PAR u.getMatrixPtr(), u.getNumElem(), value);
  // Return matrix
  return mat - u;
}

// Matrix sub with Type (LHS)
template<typename T>
const auto& operator-(const Type& value, const IMatrix<T>& mat) {
  // Create type matrix filled with value (Type)
  auto& u = CreateMatrix<Type>(mat.getNumRows(), mat.getNumColumns()); 
  std::fill_n(EXECUTION_PAR u.getMatrixPtr(), u.getNumElem(), value);
  // Return matrix
  return u - mat;
}


// Matrix sub with Type (RHS)
template<typename T>
const auto& operator-(const IMatrix<T>& mat, const Type& value) {
  // Create type matrix filled with value (Type)
  auto& u = CreateMatrix<Type>(mat.getNumRows(), mat.getNumColumns()); 
  std::fill_n(EXECUTION_PAR u.getMatrixPtr(), u.getNumElem(), value);
  // Return matrix
  return mat - u;
}