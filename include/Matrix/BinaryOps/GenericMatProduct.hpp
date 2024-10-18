/**
 * @file include/Matrix/BinaryOps/GenericMatProduct.hpp
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
#include "MatrixBasics.hpp"

// Left/right side is a Matrix
template <typename T1, typename T2, typename... Callables>
class GenericMatProduct
    : public IMatrix<GenericMatProduct<T1, T2, Callables...>> {
private:
  // Resources
  T1 *mp_left{nullptr};
  T2 *mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatProduct)
  DISABLE_MOVE(GenericMatProduct)

  // Verify dimensions of result matrix for multiplication operation
  inline constexpr bool verifyDim() const {
    // Left matrix rows
    const int lr = mp_left->getNumRows();
    // Right matrix rows
    const int rr = mp_right->getNumRows();
    // Left matrix columns
    const int lc = mp_left->getNumColumns();
    // Right matrix columns
    const int rc = mp_right->getNumColumns();
    // Condition for Matrix-Matrix multiplication
    return ((lc == rr));
  }

  // All matrices
  inline static constexpr const size_t m_size{6};
  Matrix<Type> *mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};

  // Constructor
  constexpr GenericMatProduct(T1 *u, T2 *v, Callables &&...call)
      : mp_left{u}, mp_right{v}, m_caller{std::make_tuple(
                                     std::forward<Callables>(call)...)},
        m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { return mp_left->getNumRows(); }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { return mp_right->getNumColumns(); }

  // Find me
  bool findMe(void *v) const { BINARY_FIND_ME(); }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type> *eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix multiplication dimensions mismatch");

    // Get raw pointers to result, left and right matrices
    const Matrix<Type> *left_mat = mp_left->eval();
    const Matrix<Type> *right_mat = mp_right->eval();

    // Matrix multiplication evaluation (Policy design)
    MATRIX_MUL(left_mat, right_mat, mp_arr[0]);

    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix multiplication dimensions mismatch");

    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();

    // Left and right matrices derivatives
    const Matrix<Type> *dleft_mat = mp_left->devalF(X);
    const Matrix<Type> *dright_mat = mp_right->devalF(X);

    // Left and right matrices evaluation
    const Matrix<Type> *left_mat = mp_left->eval();
    const Matrix<Type> *right_mat = mp_right->eval();

    // L (X) I - Left matrix and identity Kronocker product (Policy design)
    MATRIX_KRON(left_mat, Eye(nrows_x), mp_arr[4]);
    // R (X) I - Right matrix and identity Kronocke product (Policy design)
    MATRIX_KRON(right_mat, Eye(ncols_x), mp_arr[5]);

    // Product with left and right derivatives (Policy design)
    MATRIX_MUL(mp_arr[4], dright_mat, mp_arr[2]);
    MATRIX_MUL(dleft_mat, mp_arr[5], mp_arr[3]);

    // Addition between left and right derivatives (Policy design)
    MATRIX_ADD(mp_arr[2], mp_arr[3], mp_arr[1]);

    // Return derivative result pointer
    return mp_arr[1];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { BINARY_MAT_RESET(); }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { return "GenericMatProduct"; }

  // Destructor
  V_DTR(~GenericMatProduct()) = default;
};

// Left is Type and right is a matrix
template <typename T, typename... Callables>
class GenericMatScalarProduct
    : public IMatrix<GenericMatScalarProduct<T, Callables...>> {
private:
  // Resources
  Type m_left{};
  T *mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatScalarProduct)
  DISABLE_MOVE(GenericMatScalarProduct)

  // All matrices
  inline static constexpr const size_t m_size{2};
  Matrix<Type> *mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};

  // Constructor
  constexpr GenericMatScalarProduct(Type u, T *v, Callables &&...call)
      : m_left{u}, mp_right{v}, m_caller{std::make_tuple(
                                    std::forward<Callables>(call)...)},
        m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { return mp_right->getNumRows(); }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { return mp_right->getNumColumns(); }

  // Find me
  bool findMe(void *v) const { BINARY_RIGHT_FIND_ME(); }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type> *eval()) {
    // Get raw pointers to result and right matrices
    const Matrix<Type> *right_mat = mp_right->eval();

    // Matrix-Scalar multiplication computation (Policy design)
    MATRIX_SCALAR_MUL(m_left, right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {

    // Right matrix derivative
    const Matrix<Type> *dright_mat = mp_right->devalF(X);

    // Matrix-Scalar multiplication computation (Policy design)
    MATRIX_SCALAR_MUL(m_left, dright_mat, mp_arr[1]);

    // Return result pointer
    return mp_arr[1];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { BINARY_MAT_RIGHT_RESET(); }

  // Get type
  V_OVERRIDE(std::string_view getType() const) {
    return "GenericMatScalarProduct";
  }

  // Destructor
  V_DTR(~GenericMatScalarProduct()) = default;
};

// Left is Expression/Variable/Parameter and right is a matrix
template <typename T1, typename T2, typename... Callables>
class GenericMatScalarProductExp
    : public IMatrix<GenericMatScalarProductExp<T1, T2, Callables...>> {
private:
  // Resources
  Expression m_left;
  T2 *mp_right{nullptr};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatScalarProductExp)
  DISABLE_MOVE(GenericMatScalarProductExp)

  // All matrices
  inline static constexpr const size_t m_size{7};
  Matrix<Type> *mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};

  // Constructor
  constexpr GenericMatScalarProductExp(T1 *u, T2 *v, Callables &&...call)
      : m_left{*u}, mp_right{v}, m_caller{std::make_tuple(
                                     std::forward<Callables>(call)...)},
        m_nidx{this->m_idx_count++} {
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { return mp_right->getNumRows(); }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { return mp_right->getNumColumns(); }

  // Find me
  bool findMe(void *v) const { BINARY_RIGHT_FIND_ME(); }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type> *eval()) {
    // Get raw pointers to result and right matrices
    const Matrix<Type> *rhs = mp_right->eval();
    const Type val = Eval(m_left);

    // Matrix-Scalar addition computation (Policy design)
    MATRIX_SCALAR_MUL(val, rhs, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {
    // Left derivative and evaluation
    DevalR(m_left, X, mp_arr[2]);

    // Right matrix derivative and evaluation
    const Matrix<Type> *dright_mat = mp_right->devalF(X);
    const Matrix<Type> *right_mat = mp_right->eval();

    // Evaluate left expression
    const Type val = Eval(m_left);

    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();
    const size_t nrows_f = getNumRows();
    const size_t ncols_f = getNumColumns();

    MATRIX_KRON(Ones(nrows_f, ncols_f), mp_arr[2], mp_arr[5]);
    MATRIX_KRON(right_mat, Ones(nrows_x, ncols_x), mp_arr[6]);

    // Product with left and right derivatives (Policy design)
    MATRIX_SCALAR_MUL(val, dright_mat, mp_arr[4]);
    MATRIX_HADAMARD(mp_arr[5], mp_arr[6], mp_arr[3]);

    // Addition between left and right derivatives (Policy design)
    MATRIX_ADD(mp_arr[4], mp_arr[3], mp_arr[1]);

    // Return result pointer
    return mp_arr[1];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { BINARY_MAT_RIGHT_RESET(); }

  // Get type
  V_OVERRIDE(std::string_view getType() const) {
    return "GenericMatScalarProductExp";
  }

  // Destructor
  V_DTR(~GenericMatScalarProductExp()) = default;
};

// GenericMatProduct with 2 typename callables
template <typename T1, typename T2>
using GenericMatProductT = GenericMatProduct<T1, T2, OpMatType>;

// GenericMatScalarProduct with 1 typename and callables
template <typename T>
using GenericMatScalarProductT = GenericMatScalarProduct<T, OpMatType>;

// GenericMatScalarProductExp with 2 typename and callables
template <typename T1, typename T2>
using GenericMatScalarProductExpT =
    GenericMatScalarProductExp<T1, T2, OpMatType>;

// Function for product computation
template <typename T1, typename T2>
constexpr const auto &operator*(const IMatrix<T1> &u, const IMatrix<T2> &v) {
  auto tmp = Allocate<GenericMatProductT<T1, T2>>(
      const_cast<T1 *>(static_cast<const T1 *>(&u)),
      const_cast<T2 *>(static_cast<const T2 *>(&v)), OpMatObj);
  return *tmp;
}

// Function for product computation
template <typename T>
constexpr const auto &operator*(Type u, const IMatrix<T> &v) {
  auto tmp = Allocate<GenericMatScalarProductT<T>>(
      u, const_cast<T *>(static_cast<const T *>(&v)), OpMatObj);
  return *tmp;
}

template <typename T>
constexpr const auto &operator*(const IMatrix<T> &v, Type u) {
  return u * v;
}

// Matrix multiplication with scalar (LHS) - SFINAE'd
template <typename T1, typename T2, typename = ExpType<T1>>
constexpr const auto &operator*(const T1 &v, const IMatrix<T2> &u) {
  auto tmp = Allocate<GenericMatScalarProductExpT<T1, T2>>(
      const_cast<T1 *>(static_cast<const T1 *>(&v)),
      const_cast<T2 *>(static_cast<const T2 *>(&u)), OpMatObj);
  return *tmp;
}

// Matrix sum with scalar (RHS) - SFINAE'd
template <typename T1, typename T2, typename = ExpType<T2>>
constexpr const auto &operator*(const IMatrix<T1> &u, const T2 &v) {
  return v * u;
}