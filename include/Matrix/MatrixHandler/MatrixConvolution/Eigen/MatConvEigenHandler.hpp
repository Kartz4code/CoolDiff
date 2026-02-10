/**
 * @file include/Matrix/MatrixHandler/MatrixConvolution/Eigen/MatConvEigenHandler.hpp
 *
 * @copyright 2023-2026 Karthik Murali Madhavan Rathai
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
#include <Eigen/Dense>

// Function for 2D convolution with stride
EigenMatrix EigenConvolve2D(const EigenMatrix& input, const EigenMatrix& kernel, int stride_row, int stride_col) {
    int input_rows = input.rows();
    int input_cols = input.cols();
    int kernel_rows = kernel.rows();
    int kernel_cols = kernel.cols();

    // Calculate output dimensions
    int output_rows = ((input_rows - kernel_rows) / stride_row) + 1;
    int output_cols = ((input_cols - kernel_cols) / stride_col) + 1;

    EigenMatrix output(output_rows, output_cols);

    for (size_t r{}; r < output_rows; ++r) {
        for (size_t c{}; c < output_cols; ++c) {
            EigenMatrix input_block = input.block(r * stride_row, c * stride_col, kernel_rows, kernel_cols);
            output(r, c) = (input_block.array() * kernel.array()).sum();
        }
    }

    return output;
}

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatConvEigenHandler : public T {
    private:
        // All matrices (TODO - stateless class)
        Matrix<Type>* mp_arr{nullptr};
    
    public:
        /* Matrix-Matrix numerical convolution */
        void handle(const size_t stride_x, const size_t stride_y, const size_t pad_x, const size_t pad_y,
                    const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            // Stride must be strictly non-negative
            ASSERT(((int)stride_x > 0) && ((int)stride_y > 0), "Stride is not strictly non-negative");
            // Padding must be positive
            ASSERT(((int)pad_x >= 0) && ((int)pad_y >= 0), "Stride is not positive");

            // Assert allocator
            const auto& lhs_strategy = lhs->allocatorType();
            const auto& rhs_strategy = rhs->allocatorType();
            ASSERT((lhs_strategy == rhs_strategy), "LHS and RHS matrices are in different memory spaces");

            // Eigen handler
            EIGEN_BACKEND_HANDLER(T::handle(stride_x, stride_y, pad_x, pad_y, lhs, rhs, result), rhs_strategy);

            // Reset to zero
            CoolDiff::TensorR2::Details::ResetZero(mp_arr);

            // Convolution matrix dimensions (RHS)
            const size_t crows = rhs->getNumRows();
            const size_t ccols = rhs->getNumColumns();

            // Result matrix dimensions
            const size_t rows = (((lhs->getNumRows() + (2 * pad_x) - crows) / stride_x) + 1);
            const size_t cols = (((lhs->getNumColumns() + (2 * pad_y) - ccols) / stride_y) + 1);

            // Matrix-Matrix convolution result dimensions must be strictly non-negative
            ASSERT(((int)rows > 0 || (int)cols > 0), "Matrix-Matrix convolution dimensions invalid");

            // Pad left matrix with the required padding
            lhs->pad(pad_x, pad_y, mp_arr);

            // Padded dimensions
            const size_t prows = mp_arr->getNumRows();
            const size_t pcols = mp_arr->getNumColumns();

            // Get result matrix from pool
            MemoryManager::MatrixPool(result, rows, cols, rhs_strategy);

            // Get raw pointers to result, left and right matrices
            Type* left = const_cast<Matrix<Type>*>(mp_arr)->getMatrixPtr();
            Type* right = const_cast<Matrix<Type>*>(rhs)->getMatrixPtr();

            const Eigen::Map<EigenMatrix> left_eigen(left, prows, pcols);
            const Eigen::Map<EigenMatrix> right_eigen(right, crows, ccols);
            const auto& result_eigen = EigenConvolve2D(left_eigen, right_eigen, (int)stride_x, (int)stride_y);

            Eigen::Map<EigenMatrix>(result->getMatrixPtr(), 
                                    result_eigen.rows(), 
                                    result_eigen.cols()) = result_eigen;

            return;
        }
};