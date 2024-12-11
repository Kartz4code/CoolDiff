/**
 * @file include/Matrix/BinaryOps/GenericMatConv.hpp
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
class GenericMatConv : public IMatrix<GenericMatConv<T1, T2, Callables...>> {
private:
  // Resources
  T1 *mp_left{nullptr};
  T2 *mp_right{nullptr};

  const size_t m_stride_x{1}, m_stride_y{1};
  const size_t m_pad_x{0}, m_pad_y{0};

  // Callables
  Tuples<Callables...> m_caller;

  // Disable copy and move constructors/assignments
  DISABLE_COPY(GenericMatConv)
  DISABLE_MOVE(GenericMatConv)

  // Verify dimensions of result matrix for convolution operation
  inline constexpr bool verifyDim() const {
    // Dimension of the result of convolution operator must be strictly non-negative
    return ((int)getNumRows() > 0 || (int)getNumColumns() > 0);
  }

  // All matrices
  inline static constexpr const size_t m_size{2};
  Matrix<Type> *mp_arr[m_size]{};

public:
  // Block index
  const size_t m_nidx{};

  // Constructor
   GenericMatConv(T1 *u, T2 *v, 
                  const size_t stride_x, const size_t stride_y, 
                  const size_t pad_x, const size_t pad_y, 
                  Callables &&...call) : mp_left{u}, mp_right{v}, 
                                        m_stride_x{stride_x}, m_stride_y{stride_y},
                                        m_pad_x{pad_x}, m_pad_y{pad_y}, 
                                        m_caller{std::make_tuple(std::forward<Callables>(call)...)},
                                        m_nidx{this->m_idx_count++} {
    // Stride must be strictly non-negative
    ASSERT(((int)m_stride_x > 0) && ((int)m_stride_y > 0), "Stride is not strictly non-negative");
    // Padding must be positive
    ASSERT(((int)pad_x >= 0) && ((int)pad_y >= 0), "Stride is not positive");

    // Fill with nullptrs
    std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
  }

  // Get number of rows (After convolution)
  V_OVERRIDE(size_t getNumRows() const) {
    return (((mp_left->getNumRows() + (2 * m_pad_x) - mp_right->getNumRows()) / m_stride_x) + 1);
  }

  // Get number of columns (After convolution)
  V_OVERRIDE(size_t getNumColumns() const) {
    return (((mp_left->getNumColumns() + (2 * m_pad_y) - mp_right->getNumColumns()) / m_stride_y) + 1);
  }

  // Find me
  bool findMe(void *v) const { 
    BINARY_FIND_ME(); 
  }

  // Matrix eval computation
  V_OVERRIDE(Matrix<Type> *eval()) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix convolution dimensions invalid");

    // Get raw pointers to result, left and right matrices
    Matrix<Type> *left_mat = mp_left->eval();
    Matrix<Type> *right_mat = mp_right->eval();

    // Matrix convolution
    MATRIX_CONV(m_stride_x, m_stride_y, m_pad_x, m_pad_y, left_mat, right_mat, mp_arr[0]);

    // Return result pointer
    return mp_arr[0];
  }

  // Matrix devalF computation
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {
    // Check whether dimensions are correct
    ASSERT(verifyDim(), "Matrix-Matrix convolution dimensions invalid");

    // Derivative matrix dimensions
    const size_t nrows_x = X.getNumRows();
    const size_t ncols_x = X.getNumColumns();

    // Left and right matrices derivatives
    Matrix<Type> *dleft_mat = mp_left->devalF(X);
    Matrix<Type> *dright_mat = mp_right->devalF(X);

    // Left and right matrices evaluation
    Matrix<Type> *left_mat = mp_left->eval();
    Matrix<Type> *right_mat = mp_right->eval();

    // Matrix convolution derivative
    MATRIX_DERV_CONV(nrows_x, ncols_x, m_stride_x, m_stride_y, m_pad_x, m_pad_y,
                     left_mat, dleft_mat, right_mat, dright_mat, mp_arr[1]);

    return mp_arr[1];
  }

  // Reset visit run-time
  V_OVERRIDE(void reset()) { 
    BINARY_MAT_RESET(); 
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { 
    return "GenericMatConv"; 
  }

  // Destructor
  V_DTR(~GenericMatConv()) = default;
};

// GenericMatConv with 2 typename callables
template <typename T1, typename T2>
using GenericMatConvT = GenericMatConv<T1, T2, OpMatType>;

// Function for sub computation
template <typename T1, typename T2>
constexpr const auto &conv(const IMatrix<T1> &u, const IMatrix<T2> &v,
                           const size_t stride_x = 1, const size_t stride_y = 1,
                           const size_t pad_x = 0, const size_t pad_y = 0) {
  auto tmp = Allocate<GenericMatConvT<T1, T2>>(const_cast<T1 *>(static_cast<const T1 *>(&u)),
                                               const_cast<T2*>(static_cast<const T2 *>(&v)), 
                                               stride_x, stride_y, pad_x, pad_y, OpMatObj);
  return *tmp;
}