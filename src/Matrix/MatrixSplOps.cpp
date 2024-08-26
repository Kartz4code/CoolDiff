/**
 * @file src/Matrix/MatrixSplOps.cpp
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

#include "MatrixSplOps.hpp"

// Zero matrix addition checks
Matrix<Type>* ZeroMatAdd(Matrix<Type>* lhs, Matrix<Type>* rhs) {
    // If both lhs and rhs matrices are zero matrices
    if(lhs->getMatType() == MatrixSpl::ZEROS && 
       rhs->getMatType() == MatrixSpl::ZEROS) {
        // Technically, rhs can also be returned since both lhs, rhs are zero
        return lhs;
    }
    // If lhs is a zero matrix
    else if(lhs->getMatType() == MatrixSpl::ZEROS) {
      return rhs;
    // If rhs is a zero matrix
    } else if(rhs->getMatType() == MatrixSpl::ZEROS) {
      return lhs;
    } 
    // If neither, then return nullptr
    else {
      return nullptr;
    }
}

// Zero matrix multiplication checks
Matrix<Type>* ZeroMatMul(Matrix<Type>* lhs, Matrix<Type>* rhs) {
    // Left matrix rows
    const int lr = lhs->getNumRows();
    // Right matrix columns
    const int rc = rhs->getNumColumns();

    // If both lhs and rhs matrices are zero matrices
    if(lhs->getMatType() == MatrixSpl::ZEROS || 
       rhs->getMatType() == MatrixSpl::ZEROS) {    
        return CreateMatrixPtr<Type>(lr, rc, MatrixSpl::ZEROS);
    }
    // If neither, then return nullptr
    else {
        return nullptr;
    }
}

