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
#include "CommonMatFunctions.hpp"

// Is the matrix square
bool IsSquareMatrix(Matrix<Type>* m) {
  return (m->getNumColumns() == m->getNumRows());
}

// Is the matrix zero 
bool IsZeroMatrix(Matrix<Type>* m) {
  assert(m != nullptr && "[ERROR] Matrix is a nullptr");
  // Get Matrix pointer and number of elements
  auto *it = m->getMatrixPtr();
  const size_t n = m->getNumElem();

  // Check all elements for zero
  return std::all_of(EXECUTION_PAR 
                     it, it + n,
                     [](Type i) { return (i == (Type)(0)); });
}

// Is the matrix identity
bool IsEyeMatrix(Matrix<Type>* m) {
  // If matrix is rectangular, return false
  if (false == IsSquareMatrix(m)) {
    return false;
  }
  
  const size_t rows = m->getNumRows();
  const size_t cols = m->getNumColumns();

  for(size_t i{}; i < rows; ++i) {
    for (size_t j{}; j < cols; ++j) {
      //  Diagonal elements check 
      if((i == j) && ((*m)(i, j) != (Type)(1))) {
          return false;
      } 
      // Non-diagonal elements check
      else if((i != j) && ((*m)(i,j) != (Type)(0))) {
          return false;
        }
    }
  }
  return true;
}

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
    } else {
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
        const bool lhs_bool = IsZeroMatrix(lhs);
        const bool rhs_bool = IsZeroMatrix(rhs); 
        if(lhs_bool == true || rhs_bool == true) {
          return CreateMatrixPtr<Type>(lr, rc, MatrixSpl::ZEROS);
        } else {
          return nullptr;
        }
    }
}

// Eye matrix multiplication checks
Matrix<Type>* EyeMatMul(Matrix<Type>* lhs, Matrix<Type>* rhs) {
    // If both lhs and rhs matrices are zero matrices
    if(lhs->getMatType() == MatrixSpl::EYE && 
       rhs->getMatType() == MatrixSpl::EYE) {
        // Technically, rhs can also be returned since both lhs, rhs are zero
        return lhs;
    }
    // If lhs is a zero matrix
    else if(lhs->getMatType() == MatrixSpl::EYE) {
      return rhs;
    // If rhs is a zero matrix
    } else if(rhs->getMatType() == MatrixSpl::EYE) {
      return lhs;
    } 
    // If neither, then return nullptr
    else {
        return nullptr;
    }
}

// Zero matrix addition numerical check
Matrix<Type>* ZeroMatAddNum(Matrix<Type>* lhs, Matrix<Type>* rhs) {
    bool lhs_bool = IsZeroMatrix(lhs);
    bool rhs_bool = IsZeroMatrix(rhs);
    if(lhs_bool == true && rhs_bool == true) {
      return lhs; 
    } else if(lhs_bool == true) {
      return rhs;
    } else if (rhs_bool == true) {
      return lhs;
    } else {
      return nullptr;
    }
}

// Zero matrix multiplication numerical checks
Matrix<Type>* ZeroMatMulNum(Matrix<Type>* lhs, Matrix<Type>* rhs) {
    bool lhs_bool = IsZeroMatrix(lhs);
    bool rhs_bool = IsZeroMatrix(rhs);
    
    if(lhs_bool == true || rhs_bool == true) {
      return lhs;
    } 
    else {
      return nullptr;
    }
}

// Eye matrix multiplication numerical check
Matrix<Type>* EyeMatMulNum(Matrix<Type>* lhs, Matrix<Type>* rhs) {
    bool lhs_bool = IsEyeMatrix(lhs);
    bool rhs_bool = IsEyeMatrix(rhs);
    if(lhs_bool == true && rhs_bool == true) {
      return lhs; 
    } else if(lhs_bool == true) {
      return rhs;
    } else if (rhs_bool == true) {
      return lhs;
    } else {
      return nullptr;
    }
}

