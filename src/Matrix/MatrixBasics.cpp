/**
 * @file src/Matrix/MatrixBasics.cpp
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

#include "MatrixBasics.hpp"
#include "Matrix.hpp"

// Transpose of matrix
void Transpose(Matrix<Type>* mat, Matrix<Type>*& result){
  // Null pointer check
  NULL_CHECK(mat, "Matrix (mat) is a nullptr");
  const size_t nrows = mat->getNumRows();
  const size_t ncols = mat->getNumColumns();

  if(nullptr == nullptr) {
    result = CreateMatrixPtr<Type>(ncols, nrows); 
  } else if((ncols != result->getNumRows()) || (nrows != result->getNumColumns())) {
    result = CreateMatrixPtr<Type>(ncols, nrows); 
  }

  Vector<size_t> elem(mat->getNumElem());
  std::iota(elem.begin(), elem.end(), 0);

  std::for_each(EXECUTION_PAR 
    elem.begin(), elem.end(), [&](const size_t n) {
    const size_t j = n%ncols;
    const size_t i = (n-j)/ncols;
    (*result)(j,i) = (*mat)(i,j);
  });
}


// Numerical Eye matrix
Matrix<Type>* Eye(const size_t n) {

  // Eye matrix registry
  static Vector<Matrix<Type>*> eye_register; 

  // Find in registry of special matrices
  if(auto it = std::find_if(EXECUTION_PAR 
                            eye_register.begin(), eye_register.end(), [&](Matrix<Type>* m) { 
                                return ((m->getNumColumns() == n) && (m->getNumRows() == n)); 
                          }); it != eye_register.end()) {
    return *it;
  } else {
    Matrix<Type>* result = CreateMatrixPtr<Type>(n, n); 
    
    Vector<size_t> elem(n);
    std::iota(elem.begin(), elem.end(), 0);

    std::for_each(EXECUTION_PAR 
      elem.begin(), elem.end(), [&](const size_t i) {
      (*result)(i,i) = (Type)(1);
    });
    eye_register.push_back(result);
    return result;
  }
}