 #pragma once

/**
 * @file include/Matrix/MatrixInterface/MatrixAccessors.ipp
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

// Get matrix pointer immutable
template<typename T>
const T* Matrix<T>::getMatrixPtr() const { 
  return mp_mat; 
}

// Get matrix pointer mutable
template<typename T>
T* Matrix<T>::getMatrixPtr() { 
  return mp_mat; 
}

// Matrix 2D access using operator()() immutable
template<typename T>
const T& Matrix<T>::operator()(const size_t i, const size_t j) const {
  ASSERT((i >= 0 && i < m_rows), "Row index out of bound");
  ASSERT((j >= 0 && j < m_cols), "Column index out of bound");
  return mp_mat[i * m_cols + j];
}

// Matrix 2D access using operator()() mutable
template<typename T>
T& Matrix<T>::operator()(const size_t i, const size_t j) {
  ASSERT((i >= 0 && i < m_rows), "Row index out of bound");
  ASSERT((j >= 0 && j < m_cols), "Column index out of bound");
  return mp_mat[i * m_cols + j];
}

// Matrix 1D access using operator[] immutable
template<typename T>
const T& Matrix<T>::operator[](const size_t l) const {
  ASSERT((l >= 0 && l < getNumElem()), "Index out of bound");
  return mp_mat[l];
}

// Matrix 1D access using operator[] mutable
template<typename T>
T& Matrix<T>::operator[](const size_t l) {
  ASSERT((l >= 0 && l < getNumElem()), "Index out of bound");
  return mp_mat[l];
}

// Get a row for matrix using move semantics
template<typename T>
Matrix<T> Matrix<T>::getRow(const size_t i) && {
  ASSERT((i >= 0 && i < m_rows), "Row index out of bound");
  Matrix tmp(m_cols, 1);
  std::copy(EXECUTION_PAR mp_mat + (i * m_cols), mp_mat + ((i + 1) * m_cols), tmp.getMatrixPtr());
  return std::move(tmp);
}

// Get a row for matrix using copy semantics
template<typename T>
Matrix<T> Matrix<T>::getRow(const size_t i) const & {
  ASSERT((i >= 0 && i < m_rows), "Row index out of bound");
  Matrix tmp(m_cols, 1);
  std::copy(EXECUTION_PAR mp_mat + (i * m_cols), mp_mat + ((i + 1) * m_cols), tmp.getMatrixPtr());
  return tmp;
}

// Get a column for matrix using move semantics
template<typename T>
Matrix<T> Matrix<T>::getColumn(const size_t i) && {
  ASSERT((i >= 0 && i < m_cols), "Column index out of bound");
  Matrix tmp(m_rows, 1);

  // Iteration elements
  const auto idx = CoolDiff::Common::Range<size_t>(0, m_rows);
  // For each execution
  std::for_each(EXECUTION_PAR idx.begin(), idx.end(),
                  [this, &tmp](const size_t n) {
                  const size_t j = (n % m_cols);
                  const size_t i = (n - j) / m_cols;
                  tmp.mp_mat[j] = mp_mat[j * m_rows + i];
                  });

  return std::move(tmp);
}

// Get a column for matrix using copy semantics
template<typename T>
Matrix<T> Matrix<T>::getColumn(const size_t i) const & {
  ASSERT((i >= 0 && i < m_cols), "Column index out of bound");
  Matrix tmp(m_rows, 1);

  // Iteration elements
  const auto idx = CoolDiff::Common::Range<size_t>(0, m_rows);
  // For each execution
  std::for_each(EXECUTION_PAR idx.begin(), idx.end(),
                  [this, &tmp](const size_t n) {
                  const size_t j = (n % m_cols);
                  const size_t i = (n - j) / m_cols;
                  tmp.mp_mat[j] = mp_mat[j * m_rows + i];
                  });

  return tmp;
}

// Get number of rows
template<typename T>
size_t Matrix<T>::getNumRows() const { 
  return m_rows; 
}

// Get number of columns
template<typename T>
size_t Matrix<T>::getNumColumns() const { 
  return m_cols; 
}

// Get total number of elements
template<typename T>
size_t Matrix<T>::getNumElem() const { 
  return (getNumRows() * getNumColumns()); 
}

// Get final number of rows (for multi-layered expression)
template<typename T>
size_t Matrix<T>::getFinalNumRows() const { 
  size_t rows{m_rows}; 
  // If the type T is an Expression
  if constexpr (true == std::is_same_v<T, Expression>) {
    // If the vector of MetaVariables non-empty
    if (false == m_gh_vec.empty()) {
      // Choose the last value in this vector
      if (auto it = m_gh_vec.back(); nullptr != it) {
          if(auto* ptr = dynamic_cast<Matrix<Expression>*>(it)) {
            rows = ptr->getFinalNumRows();
          } else {
            rows = it->getNumRows();
          }
        }
      } 
    } 
  return rows;
}

// Get final number of columns (for multi-layered expression)
template<typename T>
size_t Matrix<T>::getFinalNumColumns() const { 
  size_t cols{m_cols};
  // If the type T is an Expression
  if constexpr (true == std::is_same_v<T, Expression>) {
    // If the vector of MetaVariables non-empty
    if (false == m_gh_vec.empty()) {
      // Choose the last value in this vector
      if (auto it = m_gh_vec.back(); nullptr != it) {
          if(auto* ptr = dynamic_cast<Matrix<Expression>*>(it)) {
            cols = ptr->getFinalNumColumns();
          } else {
            cols = it->getNumColumns();
          }
      }
    }
  }
  return cols;
}

// Get total final number of elements (for multi-layered expression)
template<typename T>
size_t Matrix<T>::getFinalNumElem() const {
  return (getFinalNumRows() * getFinalNumColumns());
}

// Get type of matrix
template<typename T>
MatrixSpl Matrix<T>::getMatType() const { 
  return m_type; 
}

// Get type
template<typename T>
std::string_view Matrix<T>::getType() const { 
  return "Matrix"; 
}
