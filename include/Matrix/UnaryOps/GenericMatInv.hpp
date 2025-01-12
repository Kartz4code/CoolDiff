/**
 * @file include/Matrix/UnaryOps/GenericMatInv.hpp
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

#include "Matrix.hpp"

// Left/right side is a Matrix
template <typename T, typename... Callables>
class GenericMatInv : public IMatrix<GenericMatInv<T, Callables...>> {
private:
  // Resources
  T *mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatInv)
  DISABLE_MOVE(GenericMatInv)

  // Verify dimensions of result matrix for trace operation
  inline constexpr bool verifyDim() const {
    // Right matrix rows
    const int rr = mp_right->getNumRows();
    // Right matrix columns
    const int rc = mp_right->getNumColumns();
    // Condition for square matrix for trace operation
    return (rr == rc);
  }

  // All matrices
  inline static constexpr const size_t m_size{7};
  Matrix<Type> *mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};

  // Constructor
  constexpr GenericMatInv(T *u, Callables &&...call) : mp_right{u}, 
                                                       m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                       m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { 
    return mp_right->getNumRows(); 
  }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { 
    return mp_right->getNumColumns(); 
  }

  // Find me
  bool findMe(void *v) const { 
    BINARY_RIGHT_FIND_ME(); 
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type> *eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix is not a square matrix to compute trace");

    // Get raw pointers to result and right matrices
    const Matrix<Type>* right_mat = mp_right->eval();
    const size_t n_size = mp_right->getNumRows();

    // Matrix inverse
    MATRIX_INVERSE(right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix is not a square matrix to compute trace");

    const size_t n_size = mp_right->getNumRows();
    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();

    // Right matrix derivative
    const Matrix<Type>* dright_mat = mp_right->devalF(X);
    const Matrix<Type>* right_mat = mp_right->eval();

    // Matrix inverse
    MATRIX_INVERSE(right_mat, mp_arr[1]);

    // R (X) I - Right matrix and identity Kronocke product (Policy design)
    MATRIX_KRON(mp_arr[1], Eye(nrows_x), mp_arr[2]);
    MATRIX_KRON(mp_arr[1], Eye(ncols_x), mp_arr[3]);

    MATRIX_MUL(mp_arr[2], dright_mat, mp_arr[4]);
    MATRIX_MUL(mp_arr[4], mp_arr[3], mp_arr[5]);
    MATRIX_SCALAR_MUL((Type)(-1), mp_arr[5], mp_arr[6]);

    // Return result pointer
    return mp_arr[6];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    BINARY_MAT_RIGHT_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericMatInv"; 
  }

  // Destructor
  V_DTR(~GenericMatInv()) = default;
};

// GenericMatInv with 1 typename and callables
template <typename T> 
using GenericMatInvT = GenericMatInv<T, OpMatType>;

// Function for inverse computation
template <typename T> 
constexpr const auto& inv(const IMatrix<T> &u) {
  auto tmp = Allocate<GenericMatInvT<T>>(const_cast<T*>(static_cast<const T *>(&u)), OpMatObj);
  return *tmp;
}