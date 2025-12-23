/**
 * @file include/Matrix/MatrixInterface/MatrixConstructors.ipp
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

#pragma once

#include "Matrix.hpp"

template<typename T>
void Matrix<T>::swap(Matrix& other) noexcept {
  std::swap(m_rows, other.m_rows);
  std::swap(m_cols, other.m_cols);
  std::swap(mp_mat, other.mp_mat);
  std::swap(m_gh_vec, other.m_gh_vec);
  std::swap(m_eval, other.m_eval);
  std::swap(m_devalf, other.m_devalf);
  std::swap(m_dest, other.m_dest);
  std::swap(mp_result, other.mp_result);
  std::swap(mp_dresult, other.mp_dresult);
  std::swap(m_nidx, other.m_nidx);
  std::swap(m_cache, other.m_cache);
}

template<typename T>
void Matrix<T>::assignClone(const Matrix<T>* other) {
  m_rows = other->m_rows;
  m_cols = other->m_cols;
  m_gh_vec = other->m_gh_vec;
  m_eval = other->m_eval;
  m_devalf = other->m_devalf;
  mp_result = other->mp_result;
  mp_dresult = other->mp_dresult;
  m_nidx = other->m_nidx;
  m_cache = other->m_cache;
  mp_mat = other->mp_mat;

  // The temporary that is created for assignClone is non-destroyable. 
  // It just holds the pointer of other matrix
  m_dest = false;
}

// Default constructor - Zero arguments
template<typename T>
Matrix<T>::Matrix() : m_rows{1}, 
                      m_cols{1},
                      mp_mat{new T[1]{}},
                      mp_result{nullptr}, 
                      mp_dresult{nullptr}, 
                      m_eval{false}, 
                      m_devalf{false},
                      m_dest{true},
                      m_nidx{this->m_idx_count++} {
  // Allocate CPU and GPU resources
  allocator();
}

// Constructor with rows and columns
template<typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols) : m_rows{rows}, 
                                                          m_cols{cols},
                                                          mp_result{nullptr}, 
                                                          mp_dresult{nullptr},
                                                          m_eval{false}, 
                                                          m_devalf{false},
                                                          m_dest{true},
                                                          m_nidx{this->m_idx_count++} {
  // Assert for strictly non-negative values for rows and columns
  ASSERT((rows > 0) && (cols > 0), "Row/Column size is not strictly non-negative");  
  // Allocate CPU and GPU resources
  allocator();                                                   
}

/*  Constructor with pointer stealer (When using external pointer for the matrix values, it is important
    that the resource is deleted either through RAII or manual de-allocation
*/
template<typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols, T* cpu_ptr, T* gpu_ptr)  :  m_rows{rows}, 
                                                                                    m_cols{cols},
                                                                                    mp_result{nullptr}, 
                                                                                    mp_dresult{nullptr},
                                                                                    m_eval{false}, 
                                                                                    m_devalf{false}, 
                                                                                    m_dest{false},
                                                                                    m_nidx{this->m_idx_count++} {
  // Assert for strictly non-negative values for rows and columns
  ASSERT((rows > 0) && (cols > 0), "Row/Column size is not strictly non-negative");  
  mp_mat = cpu_ptr;                                                                   
}

// Matrix clone
template<typename T>
Matrix<T>* Matrix<T>::clone(Matrix<T>*& mat) const {
  if(nullptr == mat) {
    mat = Matrix<Type>::MatrixFactory::CreateMatrixPtr(m_rows, m_cols, nullptr);
  }
  // Dont use copy assigment, due to allocation and reallocation of resources!
  mat->assignClone(this);
  return mat;
} 

 // Clone matrix expression
template<typename T>
constexpr const auto& Matrix<T>::cloneExp() const {
  return *this;
}

// Constructor with rows and columns with initial values
template<typename T>
Matrix<T>::Matrix(const size_t rows, const size_t cols, const T& val) : Matrix(rows, cols) {
  std::fill(EXECUTION_PAR mp_mat, mp_mat + getNumElem(), val);
}

// Copy constructor
template<typename T>
Matrix<T>::Matrix(const Matrix& m) :  m_rows{m.m_rows}, 
                                      m_cols{m.m_cols},
                                      m_eval{m.m_eval}, 
                                      m_devalf{m.m_devalf},
                                      m_dest{true},
                                      m_cache{m.m_cache}, 
                                      m_nidx{m.m_nidx} {
  // If T is an Expression type
  if constexpr(false == std::is_same_v<T,Expression>) {
    if(nullptr != m.mp_mat) {
      // Allocate CPU/GPU memory
      allocator();
      // Copy CPU data from argument to current matrix CPU data
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
Matrix<T>::Matrix(Matrix&& m) noexcept :  m_rows{std::exchange(m.m_rows, -1)}, 
                                          m_cols{std::exchange(m.m_cols,-1)},
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
  // Deallocate CPU/GPU memory
  if(true == m_dest) {
    deallocator();
  }
}