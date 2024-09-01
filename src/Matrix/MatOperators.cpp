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

#include "MatrixZeroOps.hpp"
#include "MatrixEyeOps.hpp"

#include "ZeroMatAddHandler.hpp"
#include "EyeMatAddHandler.hpp"
#include "ZeroMatMulHandler.hpp"
#include "EyeMatMulHandler.hpp"

#include "MatAddNaiveHandler.hpp"
#include "MatMulNaiveHandler.hpp"

// Matrix-Matrix addition - Left, Right, Result matrix pointer
void MatrixAdd(Matrix<Type>* lhs, Matrix<Type>* rhs, Matrix<Type>*& result) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Chain of responsibility (Order matters)
      1) Zero matrix check
      2) Eye matrix check 
      3) Matrix-Matrix addition
  */

  MatAddNaiveHandler h1; 
  ZeroMatAddHandler h2{&h1};
  EyeMatAddHandler h3{&h2};

  // Handle matrix addition
  h3.handle(lhs, rhs, result);
}

// Matrix-Matrix multiplication - Left, Right, Result matrix pointer
void MatrixMul(Matrix<Type>* lhs, Matrix<Type>* rhs, Matrix<Type>*& result) {
    // Null pointer check
    NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
    NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");
   
    /* Chain of responsibility (Order matters)
        1) Zero matrix check
        2) Eye matrix check 
        3) Matrix-Matrix multiplication
    */
    MatMulNaiveHandler h1;
    ZeroMatMulHandler h2{&h1};
    EyeMatMulHandler h3{&h2};

    // Handle matrix addition
    h3.handle(lhs, rhs, result);
}

// Matrix-scalar multiplication 
void MatrixScalarMul(Matrix<Type>* mat, Type val, Matrix<Type>*& result) {
    // Null pointer check
    NULL_CHECK(mat, "Matrix (mat) is a nullptr");

    // Is the matrix zero
    if(mat->getMatType() == MatrixSpl::ZEROS) {
        result = mat;
    } else {
        // Rows and columns of result matrix
        const size_t nrows{mat->getNumRows()};
        const size_t ncols{mat->getNumColumns()};    

        // If mp_result is nullptr, then create a new resource
        if (nullptr == result) {
            result = CreateMatrixPtr<Type>(nrows, ncols);
        }

        // Check for zero matrix
        if(true == IsZeroMatrix(mat)) {
            return;
        }
        // Check for identity matrix 
        else if(true == IsEyeMatrix(mat)) {
            // Element list
            Vector<size_t> elem(nrows); 
            std::iota(elem.begin(), elem.end(), 0); 
            // For each element
            std::for_each(EXECUTION_PAR 
                            elem.begin(), elem.end(), 
                            [&mat, &result, val](const size_t i) {
                            (*result)(i,i) = (*mat)(i,i)*val;
                        });
        } else {
            const size_t size{nrows*ncols};
            Type* matptr = mat->getMatrixPtr();
            Type* resptr = result->getMatrixPtr();
            std::transform(EXECUTION_PAR 
                            matptr, matptr + size, resptr, 
                            [val](const Type a) { 
                                return a*val; 
                            });
        }   
    }
}