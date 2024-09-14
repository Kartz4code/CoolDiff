/**
 * @file include/Matrix/UnaryOps/GenericMatTranspose.hpp
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

// Left/right side is a Matrix
template <typename T, typename... Callables>
class GenericMatTranspose : public IMatrix<GenericMatTranspose<T, Callables...>> {
private:
  // Resources
  T *mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatTranspose)
  DISABLE_MOVE(GenericMatTranspose)

public:
  // Result
  Matrix<Type> *mp_result{nullptr};
  // Derivative result
  Matrix<Type> *mp_dresult{nullptr};

  // Block index
  const size_t m_nidx{};

  // Constructor
  constexpr GenericMatTranspose(T *u, Callables &&...call) : mp_right{u}, 
                                                             mp_result{nullptr}, 
                                                             mp_dresult{nullptr},
                                                             m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                             m_nidx{this->m_idx_count++} 
  {}

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { 
    return  mp_right->getNumColumns();
  }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { 
    return mp_right->getNumRows();  
  }

  // Find me
  bool findMe(void *v) const { 
    BINARY_RIGHT_FIND_ME(); 
  }

  
  // Matrix eval computation
  V_OVERRIDE(Matrix<Type> *eval()) {
    // Get raw pointers to result and right matrices
    const Matrix<Type> *right_mat = mp_right->eval();

    // Matrix transpose computation (Policy design)
    MATRIX_TRANSPOSE(right_mat, mp_result);

    // Return result pointer
    return mp_result;
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {
    // Rows and columns of function and variable
    const size_t nrows_f = mp_right->getNumRows();
    const size_t ncols_f = mp_right->getNumColumns();
    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();
    
    // Result matrix dimensions
    const size_t nrows = ncols_f*nrows_x;
    const size_t ncols = nrows_f*ncols_x;

    // Right matrix derivative
    const Matrix<Type> *dright_mat = mp_right->devalF(X);
    
    if (nullptr == mp_dresult) {
        mp_dresult = CreateMatrixPtr<Type>(nrows, ncols);
    } else if ((ncols != mp_dresult->getNumRows()) ||
                (nrows != mp_dresult->getNumColumns())) {
        mp_dresult = CreateMatrixPtr<Type>(nrows, ncols);
    }

    
    for(size_t i{}; i < nrows_f; ++i) {
        for(size_t j{}; j < ncols_f; ++j) {
            for(size_t l{}; l < nrows_x; ++l) {
                for(size_t m{}; m < ncols_x; ++m) {
                    (*mp_dresult)(l+ncols_f*j, m+i*ncols_x) = (*dright_mat)(l+nrows_x*i, m+nrows_f*j);
                }
            }
        }
    }

    // Return result pointer
    return mp_dresult;
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    BINARY_MAT_RIGHT_RESET();
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) {
    return "GenericMatTranspose";
  }

  // Destructor
  V_DTR(~GenericMatTranspose()) = default;
};

// GenericMatTranspose with 1 typename and callables
template <typename T>
using GenericMatTransposeT = GenericMatTranspose<T, OpMatType>;

// Function for transpose computation
template <typename T>
constexpr const auto& transpose(const IMatrix<T> &u) {
  auto tmp = Allocate<GenericMatTransposeT<T>>(const_cast<T*>(static_cast<const T*>(&u)), OpMatObj);
  return *tmp;
}