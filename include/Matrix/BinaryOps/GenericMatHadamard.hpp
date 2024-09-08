/**
 * @file include/Matrix/BinaryOps/GenericMatHadamard.hpp
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

#include "MatrixBasics.hpp"

// Left/right side is an expression
template <typename T1, typename T2, typename... Callables>
class GenericMatHadamard
    : public IMatrix<GenericMatHadamard<T1, T2, Callables...>> {
private:
  // Resources
  T1 *mp_left{nullptr};
  T2 *mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatHadamard)
  DISABLE_MOVE(GenericMatHadamard)

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
  // Evaluaion result
  Matrix<Type> *mp_result{nullptr};

  // Derivative result
  Matrix<Type> *mp_dresult{nullptr};
  Matrix<Type> *mp_dresult_l{nullptr};
  Matrix<Type> *mp_dresult_r{nullptr};

  // Kronocker variables
  Matrix<Type> *mp_lhs_kron{nullptr};
  Matrix<Type> *mp_rhs_kron{nullptr};

  // Block index
  const size_t m_nidx{};

  // Constructor
  GenericMatHadamard(T1 *u, T2 *v, Callables &&...call)
      : mp_left{u}, mp_right{v}, mp_result{nullptr}, mp_dresult{nullptr},
        mp_dresult_l{nullptr}, mp_dresult_r{nullptr}, mp_lhs_kron{nullptr},
        mp_rhs_kron{nullptr}, m_caller{std::make_tuple(
                                  std::forward<Callables>(call)...)},
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
    ASSERT(verifyDim(), "Matrix-Matrix Hadamard product dimensions mismatch");

    // Get raw pointers to result, left and right matrices
    Matrix<Type> *left_mat = mp_left->eval();
    Matrix<Type> *right_mat = mp_right->eval();

    // Matrix-Matrix Hadamard product evaluation (Policy design)
    MATRIX_HADAMARD(left_mat, right_mat, mp_result);

    return mp_result;
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix Hadamard product dimensions mismatch");

    // Left and right matrices derivatives
    Matrix<Type> *dleft_mat = mp_left->devalF(X);
    Matrix<Type> *dright_mat = mp_right->devalF(X);

    // Left and right matrices evaluation
    Matrix<Type> *left_mat = mp_left->eval();
    Matrix<Type> *right_mat = mp_right->eval();

    // L (X) I - Left matrix and identity Kronocker product (Policy design)
    MATRIX_KRON(left_mat, Ones(X.getNumRows()), mp_lhs_kron);
    // R (X) I - Right matrix and identity Kronocke product (Policy design)
    MATRIX_KRON(right_mat, Ones(X.getNumColumns()), mp_rhs_kron);

    // Hadamard product with left and right derivatives (Policy design)
    MATRIX_HADAMARD(mp_lhs_kron, dright_mat, mp_dresult_l);
    MATRIX_HADAMARD(dleft_mat, mp_rhs_kron, mp_dresult_r);

    // Addition between left and right derivatives (Policy design)
    MATRIX_ADD(mp_dresult_l, mp_dresult_r, mp_dresult);

    // Return derivative result pointer
    return mp_dresult;
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()){BINARY_MAT_RESET()}

  // Get type
  V_OVERRIDE(std::string_view getType() const) {
    return "GenericMatHadamard";
  }

  // Destructor
  V_DTR(~GenericMatHadamard()) = default;
};

// GenericMatHadamard with 2 typename callables
template <typename T1, typename T2>
using GenericMatHadamardT = GenericMatHadamard<T1, T2, OpMatType>;

// Function for Hadamard product computation
template <typename T1, typename T2>
const GenericMatHadamardT<T1, T2> &operator^(const IMatrix<T1> &u,
                                             const IMatrix<T2> &v) {
  auto tmp = Allocate<GenericMatHadamardT<T1, T2>>(
      const_cast<T1 *>(static_cast<const T1 *>(&u)),
      const_cast<T2 *>(static_cast<const T2 *>(&v)), OpMatObj);
  return *tmp;
}