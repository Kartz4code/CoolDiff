/**
 * @file src/Matrix/Matrix.cpp
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

#include "Matrix.hpp"
#include "MemoryManager.hpp"

void CreateMatrixResource(const size_t rows, const size_t cols, Matrix<Type>*& result, Type val) {
  // Function to check for free matrices
  const auto functor = [rows,cols](const auto& m) {
    if(m != nullptr) {
      return ((m->getNumRows() == rows)    && 
              (m->getNumColumns() == cols) &&  
              (m->getMatType() == -1)      && 
              (m->m_free == true));
    } else {
      return false;
    } 
  };

  // Matrix<Type> database
  auto& mat_ptr = MemoryManager::m_del_mat_type_ptr; 

  // Dispatch matrix resource
  if (nullptr == result) {
    result = CreateMatrixPtr<Type>(rows, cols, val);
  } else if (-1 != result->getMatType()) {
    result = CreateMatrixPtr<Type>(rows, cols, val);
  } else if ((rows != result->getNumRows()) || (cols != result->getNumColumns())) {
    if(auto it = std::find_if(EXECUTION_PAR mat_ptr.begin(), mat_ptr.end(), functor); 
            it != mat_ptr.end()) {
      
      Type* ptr = (*it)->getMatrixPtr();
      const size_t nelem = (*it)->getNumElem();

      // Fill n free
      std::fill(EXECUTION_PAR ptr, ptr + nelem, val); 
      (*it)->m_free = false;

      // Result pointer 
      result = it->get();     
    } else {
      result = CreateMatrixPtr<Type>(rows, cols, val);
    }
  }
}

Matrix<Type>* DervMatrix(const size_t frows, const size_t fcols, const size_t xrows, const size_t xcols) {
    const size_t drows = frows*xrows;
    const size_t dcols = fcols*xcols;
    Matrix<Type>* dresult = CreateMatrixPtr<Type>(drows, dcols);

    // Vector of indices in X matrix
    const auto idx = Range<size_t>(0, xrows*xcols);
    // Logic for Kronecker product (With ones)
    std::for_each(EXECUTION_PAR 
                  idx.begin(), idx.end(),
                  [&](const size_t n) {
                    const size_t j = n % xcols;
                    const size_t i = (n - j) / xcols;
                    // Inner loop
                    (*dresult)(i * xrows + i, j * xcols + j) = (Type)(1);
                  });
            
    return dresult;
}