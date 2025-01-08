/**
 * @file include/Matrix/UnaryOps/GenericMatSigma.hpp
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
class GenericMatSigma : public IMatrix<GenericMatSigma<T, Callables...>> {
private:
  // Resources
  T *mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Sigma axis
  const Axis m_axis{Axis::ALL};

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatSigma)
  DISABLE_MOVE(GenericMatSigma)

  // All matrices
  inline static constexpr const size_t m_size{6};
  Matrix<Type> *mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};

  // Constructor
  constexpr GenericMatSigma(T *u, const Axis& axis, Callables &&...call) : mp_right{u}, 
                                                                           m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                                           m_axis{axis},
                                                                           m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) {
    if(m_axis == Axis::ROW) { 
      return 1;
    } else if(m_axis == Axis::COLUMN) {
      return mp_right->getNumRows();
    } else {
      return 1;
    }
  }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { 
    if(m_axis == Axis::ROW) { 
      return mp_right->getNumColumns();
    } 
    else if(m_axis == Axis::COLUMN) {
      return 1;
    } 
    else {
      return 1;
    } 
  }

  // Find me
  bool findMe(void *v) const { BINARY_RIGHT_FIND_ME(); }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type> *eval()) {
    // Get raw pointers to result and right matrices
    const Matrix<Type> *right_mat = mp_right->eval();
    const size_t rows = mp_right->getNumRows();
    const size_t cols = mp_right->getNumColumns();

    // Sum of all matrix elements as matrix operation - e_r'*A*e_c, e_{r,c}
    // being a one vector of r,c rows and columns
    if(m_axis == Axis::ROW) { 
      MATRIX_MUL(Ones(1, rows), right_mat, mp_arr[0]);
      // Return result pointer
      return mp_arr[0];
    } 
    else if(m_axis == Axis::COLUMN) {
      MATRIX_MUL(right_mat, Ones(cols, 1), mp_arr[0]);
      // Return result pointer
      return mp_arr[0];       
    } 
    else {
      MATRIX_MUL(Ones(1, rows), right_mat, mp_arr[2]);
      MATRIX_MUL(mp_arr[2], Ones(cols, 1), mp_arr[0]);
      // Return result pointer
      return mp_arr[0];
    }
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {
    // Rows and columns of function and variable
    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();
    const size_t nrows_f = mp_right->getNumRows();
    const size_t ncols_f = mp_right->getNumColumns();

    // Right matrix derivative
    const Matrix<Type> *drhs = mp_right->devalF(X);

    // Derivative of sigma function as a matrix operation
    if(m_axis == Axis::ROW) {
      MATRIX_KRON(Ones(1, nrows_f), Eye(nrows_x), mp_arr[3]);
      MATRIX_MUL(mp_arr[3], drhs, mp_arr[5]);
      // Return result pointer
      return mp_arr[5];
    }
    else if (m_axis == Axis::COLUMN) {
      MATRIX_KRON(Ones(ncols_f, 1), Eye(ncols_x), mp_arr[4]);
      MATRIX_MUL(drhs, mp_arr[4], mp_arr[1]);
      // Return result pointer
      return mp_arr[1];
    }
    else {
      MATRIX_KRON(Ones(1, nrows_f), Eye(nrows_x), mp_arr[3]);
      MATRIX_KRON(Ones(ncols_f, 1), Eye(ncols_x), mp_arr[4]);
      MATRIX_MUL(mp_arr[3], drhs, mp_arr[5]);
      MATRIX_MUL(mp_arr[5], mp_arr[4], mp_arr[1]);
      // Return result pointer
      return mp_arr[1];
    }
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { BINARY_MAT_RIGHT_RESET(); }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { return "GenericMatSigma"; }

  // Destructor
  V_DTR(~GenericMatSigma()) = default;
};

// GenericMatSigma with 1 typename and callables
template <typename T> using GenericMatSigmaT = GenericMatSigma<T, OpMatType>;

// Function for sigma computation
template <typename T> 
constexpr const auto &sigma(const IMatrix<T> &u, const Axis& axis = Axis::ALL) {
  auto tmp = Allocate<GenericMatSigmaT<T>>(
      const_cast<T*>(static_cast<const T*>(&u)), axis, OpMatObj);
  return *tmp;
}