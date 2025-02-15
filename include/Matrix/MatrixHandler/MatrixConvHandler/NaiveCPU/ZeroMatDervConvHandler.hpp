/**
 * @file
 * include/Matrix/MatrixHandler/MatrixConvHandler/NaiveCPU/ZeroMatDervConvHandler.hpp
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

 #include "MatrixStaticHandler.hpp"
 #include "Matrix.hpp"
 #include "MatrixZeroOps.hpp"
 #include "MatrixBasics.hpp"

 template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
 class ZeroMatDervConvHandler : public T {
    private:
      // All matrices
      inline static constexpr const size_t m_size{8};
      Matrix<Type> *mp_arr[m_size]{};

      // Boolean check for initialization
      bool m_initialized{false};

      void handleLHS(const size_t nrows_x, const size_t ncols_x, const size_t stride_x,
                    const size_t stride_y, const size_t pad_x, const size_t pad_y,
                    const size_t crows, const size_t ccols, const Matrix<Type> *lhs,
                    const Matrix<Type> *drhs, Matrix<Type> *&result) {
        // Stride must be strictly non-negative
        ASSERT(((int)stride_x > 0) && ((int)stride_y > 0), "Stride is not strictly non-negative");
        // Padding must be positive
        ASSERT(((int)pad_x >= 0) && ((int)pad_y >= 0), "Stride is not positive");

        // One time initialization
        if (false == m_initialized) {
          std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
          m_initialized = true;
        }

        // LHS matrix rows and columns
        const size_t lhs_rows = lhs->getNumRows();
        const size_t lhs_cols = lhs->getNumColumns();

        // Result matrix dimensions
        const size_t rows = (((lhs_rows + (2 * pad_x) - crows) / stride_x) + 1);
        const size_t cols = (((lhs_cols + (2 * pad_y) - ccols) / stride_y) + 1);

        // Matrix-Matrix convolution result dimensions must be strictly non-negative
        ASSERT(((int)rows > 0 || (int)cols > 0), "Matrix-Matrix convolution dimensions invalid");

        // Pad left matrix with required padding
        lhs->pad(pad_x, pad_y, mp_arr[0]);

        // Get result matrix from pool
        MemoryManager::MatrixPool((rows * nrows_x), (cols * ncols_x), result);

        for (size_t i{}; i < rows; ++i) {
          for (size_t j{}; j < cols; ++j) {

            // Reset to zero for the recurring matrices (#8 - #18)
            for (size_t k{1}; k <= 7; ++k) {
              ResetZero(mp_arr[k]);
            }

            // Left block matrix
            mp_arr[0]->getBlockMat({(i * stride_x), (i * stride_x) + crows - 1},
                                  {(j * stride_y), (j * stride_y) + ccols - 1},
                                  mp_arr[1]);

            // L (X) I - Left matrix and identity Kronocker product (Policy design)
            MatrixKron(mp_arr[1], Ones(nrows_x, ncols_x), mp_arr[2]);

            // Hadamard product with left and right derivatives (Policy design)
            MatrixHadamard(mp_arr[2], drhs, mp_arr[3]);

            // Sigma funcion derivative
            MatrixKron(Ones(1, crows), Eye(nrows_x), mp_arr[4]);
            MatrixKron(Ones(ccols, 1), Eye(ncols_x), mp_arr[5]);
            MatrixMul(mp_arr[4], mp_arr[3], mp_arr[6]);
            MatrixMul(mp_arr[6], mp_arr[5], mp_arr[7]);

            // Set block matrix
            result->setBlockMat({(i * nrows_x), (i + 1) * nrows_x - 1}, {(j * ncols_x), (j + 1) * ncols_x - 1}, mp_arr[7]);
          }
        }

        // Free resources
        std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [](Matrix<Type> *m) {
          if (nullptr != m) {
            m->free();
          }
        });
      }

      void handleRHS(const size_t nrows_x, const size_t ncols_x, const size_t stride_x,
                    const size_t stride_y, const size_t pad_x, const size_t pad_y,
                    const size_t lhs_rows, const size_t lhs_cols, const Matrix<Type>* rhs,
                    const Matrix<Type>* dlhs, Matrix<Type>*& result) {
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
        const size_t rows = (((lhs_rows + (2 * pad_x) - crows) / stride_x) + 1);
        const size_t cols = (((lhs_cols + (2 * pad_y) - ccols) / stride_y) + 1);

        // Matrix-Matrix convolution result dimensions must be strictly non-negative
        ASSERT(((int)rows > 0 || (int)cols > 0), "Matrix-Matrix convolution dimensions invalid");

        // Pad left derivative matrix with required padding
        dlhs->pad((pad_x * nrows_x), (pad_y * ncols_x), mp_arr[0]);

        // Get result matrix from pool
        MemoryManager::MatrixPool((rows * nrows_x), (cols * ncols_x), result);

        for (size_t i{}; i < rows; ++i) {
          for (size_t j{}; j < cols; ++j) {

            // Reset to zero for the recurring matrices (#8 - #18)
            for (size_t k{1}; k <= 7; ++k) {
              ResetZero(mp_arr[k]);
            }

            // Left derivative block matrix
            mp_arr[0]->getBlockMat({(i * stride_x * nrows_x),
                                    (i * stride_x * nrows_x) + (nrows_x * crows) - 1},
                                  {(j * stride_y * ncols_x),
                                    (j * stride_y * ncols_x) + (ncols_x * ccols) - 1},
                                  mp_arr[1]);

            // R (X) I - Right matrix and identity Kronocker product (Policy design)
            MatrixKron(rhs, Ones(nrows_x, ncols_x), mp_arr[2]);

            // Hadamard product with left and right derivatives (Policy design)
            MatrixHadamard(mp_arr[1], mp_arr[2], mp_arr[3]);

            // Sigma funcion derivative
            MatrixKron(Ones(1, crows), Eye(nrows_x), mp_arr[4]);
            MatrixKron(Ones(ccols, 1), Eye(ncols_x), mp_arr[5]);
            MatrixMul(mp_arr[4], mp_arr[3], mp_arr[6]);
            MatrixMul(mp_arr[6], mp_arr[5], mp_arr[7]);

            // Set block matrix
            result->setBlockMat({(i * nrows_x), (i + 1) * nrows_x - 1},
                                {(j * ncols_x), (j + 1) * ncols_x - 1}, mp_arr[7]);
          }
        }

        // Free resources
        std::for_each(EXECUTION_PAR mp_arr, mp_arr + m_size, [](Matrix<Type> *m) {
          if (nullptr != m) {
            m->free();
          }
        });
      }

     public:
        void handle(const size_t nrows_x, const size_t ncols_x, const size_t stride_x,
                    const size_t stride_y, const size_t pad_x, const size_t pad_y,
                    const Matrix<Type>* lhs, const Matrix<Type>* dlhs, const Matrix<Type>* rhs,
                    const Matrix<Type>* drhs, Matrix<Type>*& result) {
          // Stride must be strictly non-negative
          ASSERT(((int)stride_x > 0) && ((int)stride_y > 0), "Stride is not strictly non-negative");
          // Padding must be positive
          ASSERT(((int)pad_x >= 0) && ((int)pad_y >= 0), "Stride is not positive");

          // Convolution matrix dimensions (RHS)
          const size_t crows = rhs->getNumRows();
          const size_t ccols = rhs->getNumColumns();

          // LHS matrix rows and columns
          const size_t lhs_rows = lhs->getNumRows();
          const size_t lhs_cols = lhs->getNumColumns();

          // Result matrix dimensions
          const size_t rows = (((lhs_rows + (2 * pad_x) - crows) / stride_x) + 1);
          const size_t cols = (((lhs_cols + (2 * pad_y) - ccols) / stride_y) + 1);

          // Matrix-Matrix convolution result dimensions must be strictly non-negative
          ASSERT(((int)rows > 0 || (int)cols > 0), "Matrix-Matrix convolution dimensions invalid");

        #if defined(NAIVE_IMPL)
            /* Zero matrix special check */
            if (auto *it = ZeroMatDervConv((rows * nrows_x), (cols * ncols_x), lhs, dlhs, rhs, drhs); nullptr != it) {
              if (it == rhs) {
                handleRHS(nrows_x, ncols_x, stride_x, stride_y, pad_x, pad_y, lhs_rows, lhs_cols, rhs, dlhs, result);
                return;
              } else if (it == lhs) {
                handleLHS(nrows_x, ncols_x, stride_x, stride_y, pad_x, pad_y, crows, ccols, lhs, drhs, result);
                return;
              } else {
                result = const_cast<Matrix<Type> *>(it);
                return;
              }
              return;
            }
          /* Zero matrix numerical check */
          #if defined(NUMERICAL_CHECK)
            else if (auto *it = ZeroMatDervConvNum((rows * nrows_x), (cols * ncols_x), lhs, dlhs, rhs, drhs); nullptr != it) {
              if (it == rhs) {
                handleRHS(nrows_x, ncols_x, stride_x, stride_y, pad_x, pad_y, lhs_rows, lhs_cols, rhs, dlhs, result);
                return;
              } else if (it == lhs) {
                handleLHS(nrows_x, ncols_x, stride_x, stride_y, pad_x, pad_y, crows, ccols, lhs, drhs, result);
                return;
              } else {
                result = const_cast<Matrix<Type> *>(it);
                return;
              }
              return;
            }
          #endif
        #endif

          // Chain of responsibility
          T::handle(nrows_x, ncols_x, stride_x, stride_y, pad_x, pad_y, lhs, dlhs, rhs, drhs, result);
      }
 };