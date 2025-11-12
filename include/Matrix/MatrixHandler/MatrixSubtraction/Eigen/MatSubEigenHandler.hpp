/**
 * @file include/Matrix/MatrixHandler/MatrixSubtraction/Eigen/MatSubEigenHandler.hpp
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

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatSubEigenHandler : public T {
    public:
        void handle(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            /* Matrix-Matrix numerical subtraction */

            // Dimensions of LHS and RHS matrices
            const size_t nrows{lhs->getNumRows()};
            const size_t ncols{rhs->getNumColumns()};
            const size_t lcols{lhs->getNumColumns()};
            const size_t rrows{rhs->getNumRows()};
        
            // Assert dimensions
            ASSERT((nrows == rrows) && (ncols == lcols), "Matrix subtraction dimensions mismatch");
        
            // Pool matrix
            MemoryManager::MatrixPool(nrows, ncols, result);
        
            // Get raw pointers to result, left and right matrices
            Type* left = const_cast<Matrix<Type>*>(lhs)->getMatrixPtr();
            Type* right = const_cast<Matrix<Type>*>(rhs)->getMatrixPtr();

            const Eigen::Map<EigenMatrix> left_eigen(left, nrows, lcols);
            const Eigen::Map<EigenMatrix> right_eigen(right, rrows, ncols);
            const auto& result_eigen = left_eigen - right_eigen; 

            Eigen::Map<EigenMatrix>(result->getMatrixPtr(), 
                                    result_eigen.rows(), 
                                    result_eigen.cols()) = result_eigen;

            return;
        }
};