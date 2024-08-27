/**
 * @file src/Matrix/MatOperators.cpp
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

#include "MatOperators.hpp"
#include "Matrix.hpp"

#include "MatrixSplOps.hpp"

// Non nullptr correctness
void CheckNull(void* lhs, void* rhs) {
    assert(lhs != nullptr && "[ERROR] Left matrix is a nullptr");
    assert(rhs != nullptr && "[ERROR] Right matrix is a nullptr");
}

// Matrix-Matrix addition - Left, Right, Result matrix pointer
void MatrixAdd(Matrix<Type>* lhs, Matrix<Type>* rhs, Matrix<Type>*& result) {
    // Assert for non-null pointers 
    CheckNull(lhs, rhs);
    // Zero matrix addition check
    if(auto* it = ZeroMatAdd(lhs, rhs); nullptr != it) {
        result = it;
    } else {
        // Rows and columns of result matrix
        const size_t nrows{lhs->getNumRows()};
        const size_t ncols{rhs->getNumColumns()};    

        // If mp_result is nullptr, then create a new resource
        if (nullptr == result) {
            result = CreateMatrixPtr<Type>(nrows, ncols);
        }

       // Check numerically for lhs-rhs product is zero
        if(auto* it = ZeroMatAddNum(lhs, rhs); nullptr != it) {
            *result = *it;
        } else { 
        // Get raw pointers to result, left and right matrices
        Type *res = result->getMatrixPtr();
        Type *left = lhs->getMatrixPtr();
        Type *right = rhs->getMatrixPtr();   

        const size_t size{nrows*ncols};
        std::transform(EXECUTION_PAR 
                       left, left + size,
                       right, res, 
                       [](const Type a, const Type b) { 
                            return a + b; 
                      });
        }
    }
}

// Matrix-Matrix multiplication - Left, Right, Result matrix pointer
void MatrixMul(Matrix<Type>* lhs, Matrix<Type>* rhs, Matrix<Type>*& result) {
    // Assert for non-null pointers 
    CheckNull(lhs, rhs);
    // Zero matrix multiplication check
    if(auto* it = ZeroMatMul(lhs, rhs); nullptr != it) {
        result = it; 
    } else if(auto* it = EyeMatMul(lhs,rhs); nullptr != it) {
        result = it;
    }
    else {
        const size_t lrows = lhs->getNumRows();
        const size_t rcols = rhs->getNumColumns();
        const size_t rrows = rhs->getNumRows();

        // If mp_result is nullptr, then create a new resource
        if (nullptr == result) {
            result = CreateMatrixPtr<Type>(lrows, rcols);
        }

        // Check numerically for lhs-rhs product is identity
        if(auto* it = EyeMatMulNum(lhs, rhs); nullptr != it) {
            *result = *it;
            return;
        } else if (auto* it = ZeroMatMulNum(lhs, rhs); nullptr != it) { 
            return;
        }
        else {
            // Get raw pointers to result, left and right matrices
            Matrix<Type>& res = *result;
            Matrix<Type>& left = *lhs;
            Matrix<Type>& right = *rhs;   
            
            // Naive matrix-matrix multiplication
            Type tmp{};
            for(size_t i{}; i < lrows; ++i) {
                for(size_t j{}; j < rcols; ++j) {
                    for(size_t k{}; k < rrows; ++k) {
                        tmp += left(i,k)*right(k,j);
                    }
                    res(i,j) = tmp;
                    tmp = (Type)(0);
                }
            }    
        }
    }
}
