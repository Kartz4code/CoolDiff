/**
 * @file src/MemoryManager/MemoryManager.cpp
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

#include "MemoryManager.hpp"
#include "Matrix.hpp"

size_t MemoryManager::size() { 
  return m_del_ptr.size(); 
}

Matrix<Type>* MemoryManager::MatrixSplPool(const size_t rows, const size_t cols, const MatrixSpl& ms) {
  // Function to check for free matrices
  const auto functor = [rows, cols, ms](const auto &m) {
    if (nullptr != m) {
      return ((m->getNumRows() == rows) && (m->getNumColumns() == cols) && (m->getMatType() == ms));
    } else {
      return false;
    }
  };

  // Matrix<Type> database
  auto &mat_ptr = MemoryManager::m_del_mat_type_ptr;

  // Dispatch matrix from pool
  if (auto it = std::find_if(EXECUTION_PAR mat_ptr.begin(), mat_ptr.end(), functor); it != mat_ptr.end()) {
    return it->get();
  } else {
    return Matrix<Type>::MatrixFactory::CreateMatrixPtr(rows, cols, ms);
  }
}

void MemoryManager::MatrixPool(const size_t rows, const size_t cols, Matrix<Type>*& result, const Type& val) {
  // Function to check for free matrices
  const auto functor = [rows, cols](const auto& m) {
    if (nullptr != m) {
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

  // Dispatch matrix from pool
  if (nullptr == result) {
    result = Matrix<Type>::MatrixFactory::CreateMatrixPtr(rows, cols, val);
    result->m_free = false;
    return;
  } 
  else if ((rows != result->getNumRows())    ||
           (cols != result->getNumColumns()) || 
           (-1 != result->getMatType())) {
        if (auto it = std::find_if(EXECUTION_PAR mat_ptr.begin(), mat_ptr.end(), functor); 
                 it != mat_ptr.end()) {
          // Get underlying pointer
          Type *ptr = (*it)->getMatrixPtr();

          // Fill n free
          std::fill(EXECUTION_PAR ptr, ptr + (*it)->getNumElem(), val);
          (*it)->m_free = false;

          // Store result
          result = it->get();
          return;
        } 
        else {
          result = Matrix<Type>::MatrixFactory::CreateMatrixPtr(rows, cols, val);
          return;
        }
  } else {
    result->m_free = false;
  }
}