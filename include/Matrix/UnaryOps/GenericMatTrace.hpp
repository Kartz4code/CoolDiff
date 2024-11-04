/**
 * @file include/Matrix/UnaryOps/GenericMatTrace.hpp
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
class GenericMatTrace : public IMatrix<GenericMatTrace<T, Callables...>> {
private:
  // Resources
  T* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatTrace)
  DISABLE_MOVE(GenericMatTrace)

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
  inline static constexpr const size_t m_size{9};
  Matrix<Type> *mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};

  // Constructor
  constexpr GenericMatTrace(T* u, Callables&&...call) : mp_right{u}, 
                                                        m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                        m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { 
    return 1; 
  }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { 
    return 1; 
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

    // Hadamard product with identity matrix
    MATRIX_HADAMARD(right_mat, Eye(n_size), mp_arr[0]);

    // Sum of diagonal elements
    MATRIX_MUL(Ones(1, n_size), mp_arr[0], mp_arr[1]);
    MATRIX_MUL(mp_arr[1], Ones(n_size, 1), mp_arr[2]);

    // Return result pointer
    return mp_arr[2];
  }

  
  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix is not a square matrix to compute trace");
    
    const size_t n_size = mp_right->getNumRows();
    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();

    // Right matrix derivative
    const Matrix<Type> *dright_mat = mp_right->devalF(X);

    // L (X) I - Left matrix and identity Kronocker product (Policy design)
    MATRIX_KRON(Eye(n_size), Ones(nrows_x, ncols_x), mp_arr[3]);

    // Hadamard product with left and right derivatives (Policy design)
    MATRIX_HADAMARD(mp_arr[3], dright_mat, mp_arr[4]);

    MATRIX_KRON(Ones(1, n_size), Eye(nrows_x), mp_arr[5]);
    MATRIX_KRON(Ones(n_size, 1), Eye(ncols_x), mp_arr[6]);
    MATRIX_MUL(mp_arr[5], mp_arr[4], mp_arr[7]);
    MATRIX_MUL(mp_arr[7], mp_arr[6], mp_arr[8]);

    // Return result pointer
    return mp_arr[8];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    BINARY_MAT_RIGHT_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericMatTrace"; 
  }

  // Destructor
  V_DTR(~GenericMatTrace()) = default;
};

// GenericMatTrace with 1 typename and callables
template <typename T> 
using GenericMatTraceT = GenericMatTrace<T, OpMatType>;

// Function for trace computation
template <typename T> 
constexpr const auto& trace(const IMatrix<T> &u) {
  auto tmp = Allocate<GenericMatTraceT<T>>(const_cast<T*>(static_cast<const T*>(&u)), OpMatObj);
  return *tmp;
}