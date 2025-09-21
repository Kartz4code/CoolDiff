 #pragma once

/**
 * @file include/Matrix/MatrixInterface/MatrixConstructors.ipp
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
 
template<typename T>
void Matrix<T>::swap(Matrix& other) noexcept {
  std::swap(m_rows, other.m_rows);
  std::swap(m_cols, other.m_cols);
  std::swap(m_type, other.m_type);
  std::swap(mp_mat, other.mp_mat);
  std::swap(m_gh_vec, other.m_gh_vec);
  std::swap(m_free, other.m_free);
  std::swap(m_eval, other.m_eval);
  std::swap(m_devalf, other.m_devalf);
  std::swap(m_dest, other.m_dest);
  std::swap(mp_result, other.mp_result);
  std::swap(mp_dresult, other.mp_dresult);
  std::swap(m_nidx, other.m_nidx);
  std::swap(m_cache, other.m_cache);
}

// Special matrix constructor (Privatized, only for internal factory view)
template<typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols, const MatrixSpl& type) : m_rows{rows}, 
                                                                                 m_cols{cols},
                                                                                 m_type{type} {
  // Assert for strictly non-negative values for rows and columns
  ASSERT((rows > 0) && (cols > 0), "Row/Column size is not strictly non-negative");                                                                                  
}

// Default constructor - Zero arguments
template<typename T>
Matrix<T>::Matrix() : m_rows{1}, 
                      m_cols{1}, 
                      m_type{(size_t)(-1)}, 
                      mp_mat{new T[1]{}},
                      mp_result{nullptr}, 
                      mp_dresult{nullptr}, 
                      m_eval{false}, 
                      m_devalf{false},
                      m_dest{true},
                      m_nidx{this->m_idx_count++} 
{}


// Default constructor - For cloning 
template<typename T>
Matrix<T>::Matrix(void*) {}

// Constructor with rows and columns
template<typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols) : m_rows{rows}, 
                                                          m_cols{cols}, 
                                                          m_type{(size_t)(-1)},
                                                          mp_result{nullptr}, 
                                                          mp_dresult{nullptr},
                                                          m_eval{false}, 
                                                          m_devalf{false},
                                                          m_dest{true},
                                                          m_nidx{this->m_idx_count++} {
  // Assert for strictly non-negative values for rows and columns
  ASSERT((rows > 0) && (cols > 0), "Row/Column size is not strictly non-negative");  
  mp_mat = new T[getNumElem()]{};                                                      
}

// Constructor with pointer stealer
template<typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols, T* ptr)  :    m_rows{rows}, 
                                                                      m_cols{cols}, 
                                                                      m_type{(size_t)(-1)},
                                                                      mp_result{nullptr}, 
                                                                      mp_dresult{nullptr},
                                                                      m_eval{false}, 
                                                                      m_devalf{false}, 
                                                                      m_dest{false},
                                                                      m_nidx{this->m_idx_count++} {
  // Assert for strictly non-negative values for rows and columns
  ASSERT((rows > 0) && (cols > 0), "Row/Column size is not strictly non-negative");  
  mp_mat = ptr;                                                                     
}

// Matrix clone
template<typename T>
Matrix<T>* Matrix<T>::clone(Matrix<T>*& mat) const {
  MemoryManager::MatrixPool(m_rows, m_cols, mat);
  *mat = *this;
  return mat;
} 

// Constructor with rows and columns with initial values
template<typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols, const T& val) : Matrix(rows, cols) {
  static_assert(is_numeric_v<T> == true, "Type of matrix is not numeric");
  std::fill(EXECUTION_PAR mp_mat, mp_mat + getNumElem(), val);
}

// Copy constructor
template<typename T>
Matrix<T>::Matrix(const Matrix& m) : m_rows{m.m_rows}, 
                                     m_cols{m.m_cols}, 
                                     m_type{m.m_type},
                                     m_eval{m.m_eval}, 
                                     m_devalf{m.m_devalf},
                                     m_dest{m.m_dest},
                                     m_cache{m.m_cache}, 
                                     m_nidx{m.m_nidx} {
  // If T is an Expression type
  if constexpr(false == std::is_same_v<T,Expression>) {
    if(nullptr != m.mp_mat) {
      // Copy values
      mp_mat = new T[getNumElem()]{};
      std::copy(EXECUTION_PAR m.mp_mat, m.mp_mat + getNumElem(), mp_mat);
    }
  } else {
      // Pushback the expression in a generic holder
      m_gh_vec.push_back((Matrix<Expression>*)&m);
  }
}

// Copy assignment operator
template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix& m) {
  // Copy-swap idiom
  Matrix<T>{m}.swap(*this);
  // Return this reference
  return *this;
}

// Move constructor
template<typename T>
Matrix<T>::Matrix(Matrix&& m) noexcept : m_rows{std::exchange(m.m_rows, -1)}, 
                                         m_cols{std::exchange(m.m_cols,-1)},
                                         m_type{std::exchange(m.m_type, -1)}, 
                                         mp_mat{std::exchange(m.mp_mat, nullptr)},
                                         mp_result{std::exchange(m.mp_result, nullptr)},
                                         mp_dresult{std::exchange(m.mp_dresult, nullptr)}, 
                                         m_eval{std::exchange(m.m_eval, false)},
                                         m_devalf{std::exchange(m.m_devalf, false)}, 
                                         m_dest{std::exchange(m.m_dest, true)},
                                         m_cache{std::move(m.m_cache)},
                                         m_gh_vec{std::exchange(m.m_gh_vec, {})}, 
                                         m_nidx{std::exchange(m.m_nidx, -1)} 
{}


// Move assignment operator
template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& m) noexcept {
    // Copy-swap idiom
    Matrix<T>{std::move(m)}.swap(*this);
    // Return this reference
    return *this;
}

// Destructor
template<typename T>
Matrix<T>::~Matrix() {
  // If mp_mat is not nullptr, delete it
  if(true == m_dest) {
    if (nullptr != mp_mat) {
      delete[] mp_mat;
      mp_mat = nullptr;
    }
  }
}