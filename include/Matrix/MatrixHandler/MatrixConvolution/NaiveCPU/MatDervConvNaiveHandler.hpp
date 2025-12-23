/**
 * @file
 * include/Matrix/MatrixHandler/MatrixConvolution/NaiveCPU/MatDervConvNaiveHandler.hpp
 *
 * @copyright 2023-2025 Karthik Murali Madhavan Rathai
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

#include "MatrixStaticHandler.hpp"
#include "MatOperators.hpp"
#include "Matrix.hpp"
#include "MatrixBasics.hpp"

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatDervConvNaiveHandler : public T {
  private:
    // All matrices (TODO - stateless class)
    inline static constexpr const size_t m_size{13};
    Matrix<Type>* mp_arr[m_size]{};

    // Boolean check for initialization
    bool m_initialized{false};
    
  public:
    void handle(const size_t nrows_x, const size_t ncols_x, const size_t stride_x, const size_t stride_y, const size_t pad_x, const size_t pad_y,
                const Matrix<Type>* lhs, const Matrix<Type>* dlhs, const Matrix<Type>* rhs, const Matrix<Type>* drhs, Matrix<Type>*& result) {
      // Stride must be strictly non-negative
      ASSERT(((int)stride_x > 0) && ((int)stride_y > 0), "Stride is not strictly non-negative");
      // Padding must be positive
      ASSERT(((int)pad_x >= 0) && ((int)pad_y >= 0), "Stride is not positive");

      // One time initialization
      if (false == m_initialized) {
        std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
        m_initialized = true;
      }

      // Convolution matrix dimensions (RHS)
      const size_t crows = rhs->getNumRows();
      const size_t ccols = rhs->getNumColumns();

      // Result matrix dimensions
      const size_t rows = (((lhs->getNumRows() + (2 * pad_x) - crows) / stride_x) + 1);
      const size_t cols = (((lhs->getNumColumns() + (2 * pad_y) - ccols) / stride_y) + 1);

      // Matrix-Matrix convolution result dimensions must be strictly non-negative
      ASSERT(((int)rows > 0 || (int)cols > 0), "Matrix-Matrix convolution dimensions invalid");

      // Pad left derivative matrix with required padding
      dlhs->pad((pad_x * nrows_x), (pad_y * ncols_x), mp_arr[0]);
      // Pad left matrix with required padding
      lhs->pad(pad_x, pad_y, mp_arr[1]);

      // Get result matrix from pool
      MemoryManager::MatrixPool(result, (rows * nrows_x), (cols * ncols_x));

      for (size_t i{}; i < rows; ++i) {
        for (size_t j{}; j < cols; ++j) {

          // Reset to zero for the recurring matrices (#8 - #18)
          for (size_t k{2}; k <= 12; ++k) {
            CoolDiff::TensorR2::Details::ResetZero(mp_arr[k]);
          }

          // Left block matrix
          mp_arr[1]->getBlockMat( {(i * stride_x), (i * stride_x) + crows - 1},
                                  {(j * stride_y), (j * stride_y) + ccols - 1},
                                  mp_arr[2] );

          // Left derivative block matrix
          mp_arr[0]->getBlockMat( {(i * stride_x * nrows_x), (i * stride_x * nrows_x) + (nrows_x * crows) - 1},
                                  {(j * stride_y * ncols_x), (j * stride_y * ncols_x) + (ncols_x * ccols) - 1},
                                  mp_arr[3] );

          // L (X) I - Left matrix and identity Kronocker product (Policy design)
          CoolDiff::TensorR2::MatOperators::MatrixKron(mp_arr[2], CoolDiff::TensorR2::MatrixBasics::Ones(nrows_x, ncols_x), mp_arr[4]);
          // R (X) I - Right matrix and identity Kronocker product (Policy design)
          CoolDiff::TensorR2::MatOperators::MatrixKron(rhs, CoolDiff::TensorR2::MatrixBasics::Ones(nrows_x, ncols_x), mp_arr[5]);

          // Hadamard product with left and right derivatives (Policy design)
          CoolDiff::TensorR2::MatOperators::MatrixHadamard(mp_arr[4], drhs, mp_arr[6]);
          CoolDiff::TensorR2::MatOperators::MatrixHadamard(mp_arr[3], mp_arr[5], mp_arr[7]);

          // Addition between left and right derivatives (Policy design)
          CoolDiff::TensorR2::MatOperators::MatrixAdd(mp_arr[6], mp_arr[7], mp_arr[8]);

          // Sigma funcion derivative
          CoolDiff::TensorR2::MatOperators::MatrixKron( CoolDiff::TensorR2::MatrixBasics::Ones(1, crows), 
                                                        CoolDiff::TensorR2::MatrixBasics::Eye(nrows_x), 
                                                        mp_arr[9] );

          CoolDiff::TensorR2::MatOperators::MatrixKron( CoolDiff::TensorR2::MatrixBasics::Ones(ccols, 1), 
                                                        CoolDiff::TensorR2::MatrixBasics::Eye(ncols_x), 
                                                        mp_arr[10]  );

          CoolDiff::TensorR2::MatOperators::MatrixMul(mp_arr[9], mp_arr[8], mp_arr[11]);
          CoolDiff::TensorR2::MatOperators::MatrixMul(mp_arr[11], mp_arr[10], mp_arr[12]);

          // Set block matrix
          result->setBlockMat({(i * nrows_x), (i + 1) * nrows_x - 1},
                              {(j * ncols_x), (j + 1) * ncols_x - 1}, 
                              mp_arr[12]);
        }
      }
    }
};