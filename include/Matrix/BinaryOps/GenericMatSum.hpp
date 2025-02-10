/**
 * @file include/Matrix/BinaryOps/GenericMatSum.hpp
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
#include "MatrixHelper.hpp"

// Left/right side is a Matrix
template <typename T1, typename T2, typename... Callables>
class GenericMatSum : public IMatrix<GenericMatSum<T1, T2, Callables...>> {
private:
  // Resources
  T1* mp_left{nullptr};
  T2* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatSum)
  DISABLE_MOVE(GenericMatSum)

  // Verify dimensions of result matrix for addition operation
  inline constexpr bool verifyDim() const {
    // Left matrix rows and columns
    const size_t lr = mp_left->getNumRows();
    const size_t lc = mp_left->getNumColumns();

    // Right matrix rows and columns
    const size_t rr = mp_right->getNumRows();
    const size_t rc = mp_right->getNumColumns();

    // Condition for Matrix-Matrix addition
    return ((lr == rr) && (lc == rc));
  }

  // All matrices
  inline static constexpr const size_t m_size{2};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};
  // Cache for reverse AD 1st
  OMMatPair m_cache{};

  // Constructor
  constexpr GenericMatSum(T1* u, T2* v, Callables&&... call) : mp_left{u}, 
                                                               mp_right{v}, 
                                                               m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                               m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { return mp_left->getNumRows(); }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { return mp_right->getNumColumns(); }

  // Find me
  bool findMe(void* v) const { BINARY_FIND_ME(); }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix addition dimensions mismatch");

    // Get raw pointers to result, left and right matrices
    const Matrix<Type>* left_mat = mp_left->eval();
    const Matrix<Type>* right_mat = mp_right->eval();

    // Matrix-Matrix addition computation (Policy design)
    MATRIX_ADD(left_mat, right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix addition dimensions mismatch");

    // Left and right matrices
    const Matrix<Type>* dleft_mat = mp_left->devalF(X);
    const Matrix<Type>* dright_mat = mp_right->devalF(X);

    // Matrix-Matrix derivative addition computation (Policy design)
    MATRIX_ADD(dleft_mat, dright_mat, mp_arr[1]);

    // Return result pointer
    return mp_arr[1];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { BINARY_MAT_RESET(); }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { return "GenericMatSum"; }

  // Destructor
  V_DTR(~GenericMatSum()) = default;
};

// Left is Type (scalar) and right is a matrix
template <typename T, typename... Callables>
class GenericMatScalarSum : public IMatrix<GenericMatScalarSum<T, Callables...>> {
private:
  // Resources
  Type m_left{};
  T* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatScalarSum)
  DISABLE_MOVE(GenericMatScalarSum)

  // All matrices
  inline static constexpr const size_t m_size{2};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};

  // Constructor
  constexpr GenericMatScalarSum(Type u, T* v, Callables&&... call) : m_left{u}, 
                                                                     mp_right{v}, 
                                                                     m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                                                     m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { return mp_right->getNumRows(); }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { return mp_right->getNumColumns(); }

  // Find me
  bool findMe(void* v) const { BINARY_RIGHT_FIND_ME(); }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Get raw pointers to result and right matrices
    const Matrix<Type>* right_mat = mp_right->eval();

    // Matrix-Scalar addition computation (Policy design)
    MATRIX_SCALAR_ADD(m_left, right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Right matrix derivative
    mp_arr[1] = mp_right->devalF(X);
    // Return result pointer
    return mp_arr[1];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { BINARY_MAT_RIGHT_RESET(); }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { return "GenericMatScalarSum"; }

  // Destructor
  V_DTR(~GenericMatScalarSum()) = default;
};

// Left is Expression/Variable/Parameter (scalar) and right is a matrix
template <typename T1, typename T2, typename... Callables>
class GenericMatScalarSumExp : public IMatrix<GenericMatScalarSumExp<T1, T2, Callables...>> {
private:
  // Resources
  T1* mp_left{nullptr};
  T2* mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatScalarSumExp)
  DISABLE_MOVE(GenericMatScalarSumExp)

  // All matrices
  inline static constexpr const size_t m_size{4};
  Matrix<Type>* mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};

  // Constructor
  constexpr GenericMatScalarSumExp(T1* u, T2* v, Callables&&... call) : mp_left{u}, 
                                                                        mp_right{v}, 
                                                                        m_caller{std::make_tuple(
                                                                        std::forward<Callables>(call)...)},
                                                                        m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { return mp_right->getNumRows(); }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { return mp_right->getNumColumns(); }

  // Find me
  bool findMe(void* v) const { BINARY_RIGHT_FIND_ME(); }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type>* eval()) {
    // Get raw pointers to result and right matrices
    const Matrix<Type>* rhs = mp_right->eval();
    const Type val = Eval((*mp_left));

    // Matrix-Scalar addition computation (Policy design)
    MATRIX_SCALAR_ADD(val, rhs, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
    // Right matrix derivative
    const Matrix<Type>* drhs = mp_right->devalF(X);

    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();
    const size_t nrows_f = getNumRows();
    const size_t ncols_f = getNumColumns();

    // Derivative of expression w.r.t to variable matrix (Reverse mode)
    DevalR((*mp_left), X, mp_arr[2]);

    // Kronecker product with ones and add with right derivatives
    MATRIX_KRON(Ones(nrows_f, ncols_f), mp_arr[2], mp_arr[3]);
    MATRIX_ADD(mp_arr[3], drhs, mp_arr[1]);

    // Return result pointer
    return mp_arr[1];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { BINARY_MAT_RIGHT_RESET(); }

  // Get type
  V_OVERRIDE(std::string_view getType() const) {
    return "GenericMatScalarSumExp";
  }

  // Destructor
  V_DTR(~GenericMatScalarSumExp()) = default;
};

// GenericMatSum with 2 typename and callables
template <typename T1, typename T2>
using GenericMatSumT = GenericMatSum<T1, T2, OpMatType>;

// GenericMatScalarSum with 1 typename and callables
template <typename T>
using GenericMatScalarSumT = GenericMatScalarSum<T, OpMatType>;

// GenericMatScalarSumExp with 2 typename and callables
template <typename T1, typename T2>
using GenericMatScalarSumExpT = GenericMatScalarSumExp<T1, T2, OpMatType>;

// Function for sum computation
template <typename T1, typename T2>
constexpr const auto& operator+(const IMatrix<T1>& u, const IMatrix<T2>& v) {
  auto tmp = Allocate<GenericMatSumT<T1, T2>>(const_cast<T1 *>(static_cast<const T1*>(&u)),
                                              const_cast<T2 *>(static_cast<const T2*>(&v)), 
                                              OpMatObj);
  return *tmp;
}

// Function for sum computation
template <typename T>
constexpr const auto& operator+(Type u, const IMatrix<T>& v) {
  auto tmp = Allocate<GenericMatScalarSumT<T>>(u, const_cast<T*>(static_cast<const T*>(&v)), OpMatObj);
  return *tmp;
}

template <typename T>
constexpr const auto &operator+(const IMatrix<T>& v, Type u) {
  return u + v;
}

// Matrix sum with scalar (LHS) - SFINAE'd
template <typename T1, typename T2, typename = ExpType<T1>>
constexpr const auto &operator+(const T1& v, const IMatrix<T2>& u) {
  auto tmp = Allocate<GenericMatScalarSumExpT<T1, T2>>(const_cast<T1*>(static_cast<const T1*>(&v)),
                                                       const_cast<T2*>(static_cast<const T2*>(&u)), 
                                                       OpMatObj);
  return *tmp;
}

// Matrix sum with scalar (RHS) - SFINAE'd
template <typename T1, typename T2, typename = ExpType<T2>>
constexpr const auto &operator+(const IMatrix<T1>& u, const T2& v) {
  return v + u;
}