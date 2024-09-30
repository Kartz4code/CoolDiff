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

#include "Matrix.hpp"

// Left/right side is a Matrix
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

  // All matrices
  inline static constexpr const size_t m_size{2};
  Matrix<Type>* mp_arr[m_size]{}; 

public:
  // Block index
  const size_t m_nidx{};

  // Constructor
  constexpr GenericMatSub(T1 *u, T2 *v, Callables &&...call) : mp_left{u}, 
                                                               mp_right{v},
                                                               m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                               m_nidx{this->m_idx_count++} {
    std::fill_n(mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { 
    return mp_left->getNumRows(); 
  }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { 
    return mp_right->getNumColumns(); 
  }

  // Find me
  bool findMe(void *v) const { 
    BINARY_FIND_ME(); 
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type> *eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix subtraction dimensions mismatch");

    // Get raw pointers to result, left and right matrices
    const Matrix<Type> *left_mat = mp_left->eval();
    const Matrix<Type> *right_mat = mp_right->eval();

    // Matrix-Matrix subtraction computation (Policy design)
    MATRIX_SUB(left_mat, right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix subtraction dimensions mismatch");

    // Left and right matrices
    const Matrix<Type> *dleft_mat = mp_left->devalF(X);
    const Matrix<Type> *dright_mat = mp_right->devalF(X);

    // Matrix-Matrix derivative subtraction computation (Policy design)
    MATRIX_SUB(dleft_mat, dright_mat, mp_arr[1]);

    // Return result pointer
    return mp_arr[1];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()){ 
    BINARY_MAT_RESET();
  }

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
constexpr const auto &operator-(const IMatrix<T1> &u,
                                const IMatrix<T2> &v) {
  auto tmp = Allocate<GenericMatSubT<T1, T2>>(
      const_cast<T1 *>(static_cast<const T1 *>(&u)),
      const_cast<T2 *>(static_cast<const T2 *>(&v)), OpMatObj);
  return *tmp;
}

// Matrix sub with Type (LHS)
template <typename T>
constexpr const auto &operator-(const Type &v, const IMatrix<T> &u) {
  return (v + ((Type)(-1)*u));
}

// Matrix sub with Type (RHS)
template <typename T>
constexpr const auto &operator-(const IMatrix<T> &u, const Type &v) {
  return (u + ((Type)(-1)*v));  
}

// Matrix sub with scalar (LHS) - SFINAE'd
template <typename T1, typename T2, typename = ExpType<T1>>
constexpr const auto &operator-(const T1 &v, const IMatrix<T2> &u) {
  return (v + ((Type)(-1)*u));
}

// Matrix sub with scalar (RHS) - SFINAE'd
template <typename T1, typename T2, typename = ExpType<T2>>
constexpr const auto &operator-(const IMatrix<T1> &u, const T2 &v) {
  return (u + ((Type)(-1)*v));
}
