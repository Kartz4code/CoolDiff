/**
 * @file src/MemoryManager/MemoryManager.cpp
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

#include "MemoryManager.hpp"
#include "Matrix.hpp"

const size_t MemoryManager::size() { 
  return m_del_ptr.size(); 
}

void MemoryManager::MatrixPool(const size_t rows, const size_t cols, Matrix<Type>*& result, const Type& val) {
  // Dispatch matrix from pool
  if (nullptr == result) {
    result = Matrix<Type>::MatrixFactory::CreateMatrixPtr(rows, cols, val);
    return;
  } 
  else if ((rows != result->getNumRows()) || (cols != result->getNumColumns())) {
    result = Matrix<Type>::MatrixFactory::CreateMatrixPtr(rows, cols, val);
    return;
  } else {
    // Never reset result to zero. Some operations may take the same input and output
    // e.g. MatrixAdd(result, A, result)
    return;
  }
}