/**
 * @file include/Matrix/Matrix.hpp
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

#include "CommonFunctions.hpp"
#include "IMatrix.hpp"
#include "MatrixBasics.hpp"
//#include <cuda_runtime.h>

// Derivative of matrices (Reverse AD)
Matrix<Type>* DervMatrix(const size_t, const size_t, const size_t, const size_t);

// Matrix class
template <typename T> 
class Matrix : public IMatrix<Matrix<T>> {
public:
  // Matrix factory
  class MatrixFactory {
  public:
    // Create matrix as reference
    template <typename... Args> 
    static Matrix<T>& CreateMatrix(Args&&... args) {
      auto tmp = Allocate<Matrix<T>>(std::forward<Args>(args)...);
      return *tmp;
    }

    // Create matrix as pointer
    template <typename... Args>
    static Matrix<T>* CreateMatrixPtr(Args&&... args) {
      auto tmp = Allocate<Matrix<T>>(std::forward<Args>(args)...);
      return tmp.get();
    }
  };

private:
  // Swap for assignment
  void swap(Matrix&) noexcept;

  // Assign clone
  void assignClone(const Matrix*);

  // Allocate friend function
  template <typename Z, typename... Argz>
  friend SharedPtr<Z> Allocate(Argz&&...);

  // Memory manager is a friend class
  friend class MemoryManager;

  // Matrices of template types is a friend class
  template <typename Z> 
  friend class Matrix;

  // Matrix row and column size
  size_t m_rows{0};
  size_t m_cols{0};

  // Matrix raw pointer of underlying type (Expression, Variable, Parameter, Type)
  T* mp_mat{nullptr};

  // Collection of meta variable expressions
  Vector<MetaMatrix*> m_gh_vec{};

private:
  // Boolean to verify evaluation/forward derivative values
  bool m_eval{false};
  bool m_devalf{false};
  
  // Should destructor be called? (True by default)
  bool m_dest{true};

  // Matrix pointer for evaluation result (Type)
  Matrix<Type>* mp_result{nullptr};
  // Matrix pointer for forward derivative (Type)
  Matrix<Type>* mp_dresult{nullptr};

private:  
  // Set values for the result matrix
  void setEval();

  // Set value for the derivative result matrix
  void setDevalF(const Matrix<Variable>&);

public:
  // Block index
  size_t m_nidx{};

  // Cache for reverse AD
  OMMatPair m_cache{};

  // Default constructor - Zero arguments
  Matrix();
  
  // Constructor with rows and columns
  Matrix(const size_t, const size_t);
  
  // Constructor with rows and columns with initial values
  Matrix(const size_t, const size_t, const T&);

  // Constructor with pointer stealer
  Matrix(const size_t, const size_t, T*);

  // Matrix clone
  Matrix* clone(Matrix*&) const;

  // Clone matrix expression
  constexpr const auto& cloneExp() const;

  // Matrix reshape
  void reshape(const size_t, const size_t);

  // Matrix expression constructor
  template <typename Z>
  Matrix(const IMatrix<Z>& expr) :  m_rows{expr.getNumRows()}, 
                                    m_cols{expr.getNumColumns()},
                                    mp_mat{nullptr}, 
                                    mp_result{nullptr},
                                    mp_dresult{nullptr}, 
                                    m_eval{false}, 
                                    m_devalf{false},
                                    m_dest{true},
                                    m_nidx{this->m_idx_count++} {
    // Static assert so that type T is an expression
    static_assert(true == std::is_same_v<T, Expression>, "[ERROR] The type T is not an expression");
    // Reserve a buffer of Matrix expressions
    m_gh_vec.reserve(g_vec_init);
    // Expression multiplied with one and emplace it in a generic holder
    m_gh_vec.push_back((Matrix<Expression>*)&(expr*1));
  }

  /* Copy assignment for expression evaluation */
  template <typename Z> 
  Matrix& operator=(const IMatrix<Z>& expr) {
    // Static assert so that type T is an expression
    static_assert(true == std::is_same_v<T, Expression>, "[ERROR] The type T is not an expression");
    // Clear buffer and set rows and columns if not recursive expression not found
    if (static_cast<const Z&>(expr).findMe(this) == false) {
      m_gh_vec.clear();
    }
    // If the push back bector is zero
    if (true == m_gh_vec.empty()) {
      m_rows = expr.getNumRows();
      m_cols = expr.getNumColumns();
      if (nullptr != mp_mat) {
        delete[] mp_mat;
        mp_mat = nullptr;
      }
      if (nullptr != mp_result) {
        delete[] mp_result;
        mp_result = nullptr;
      }
      if (nullptr != mp_dresult) {
        delete[] mp_dresult;
        mp_dresult = nullptr;
      }
      m_eval = false;
      m_devalf = false;
      m_dest = true;
    }
    // Expression multiplied with one and emplace it in a generic holder
    m_gh_vec.push_back((Matrix<Expression>*)&(expr*1));
    return *this;
  }

  // Copy constructor
  Matrix(const Matrix&);
  
  // Copy assignment operator
  Matrix& operator=(const Matrix&);

  // Move constructor
  Matrix(Matrix&&) noexcept;
  
  // Move assignment operator
  Matrix& operator=(Matrix&&) noexcept;

  // Get matrix pointer immutable
  const T* getMatrixPtr() const;

  // Get matrix pointer mutable
  T* getMatrixPtr();

  // Set matrix pointer
  void setMatrixPtr(T*);
  
  // Matrix 2D access using operator()() immutable
  const T& operator()(const size_t, const size_t) const;

  // Matrix 2D access using operator()() mutable
  T& operator()(const size_t, const size_t);

  // Matrix 1D access using operator[] immutable
  const T& operator[](const size_t) const;
  
  // Matrix 1D access using operator[] mutable
  T& operator[](const size_t);

  // Get block matrix
  void getBlockMat(const Pair<size_t, size_t>&, const Pair<size_t, size_t>&, Matrix*&) const;

  // Set block matrix
  void setBlockMat(const Pair<size_t, size_t>&, const Pair<size_t, size_t>&, const Matrix*);

  // Copy data from another matrix (Just copy all contents from one matrix to another)
  void copyData(const Matrix<T>&);

  // Copy data from a pointer
  void copyData(T*);

  // Add zero padding
  void pad(const size_t, const size_t, Matrix*&) const;

  // Get a row for matrix using move semantics
  Matrix getRow(const size_t) &&;

  // Get a row for matrix using copy semantics
  Matrix getRow(const size_t) const &;

  // Get row pointer (Default ordering is row major)
  T* getRowPtr(const size_t i) const;

  // Get a column for matrix using move semantics
  Matrix getColumn(const size_t) &&;

  // Get a column for matrix using copy semantics
  Matrix getColumn(const size_t) const &;

  // Get total elements
  size_t getNumElem() const;

  // Get final number of rows (for multi-layered expression)
  size_t getFinalNumRows() const;
  
  // Get final number of columns (for multi-layered expression)
  size_t getFinalNumColumns() const;

  // Get total final number of elements (for multi-layered expression)
  size_t getFinalNumElem() const;

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const);
  
  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const);

  // Evaluate matrix
  V_OVERRIDE(Matrix<Type>* eval());

  // Derivative matrix
  V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>&));

  // Traverse
  V_OVERRIDE(void traverse(OMMatPair* cache = nullptr));

  // Get cache
  V_OVERRIDE(OMMatPair& getCache());
  
  // Reset all visited flags
  V_OVERRIDE(void reset());

  // Get type
  V_OVERRIDE(std::string_view getType() const);
  
  // Find me (Expression finder)
  bool findMe(void*) const;

  // Reset impl (Expression resetter)
  void resetImpl();

  // To output stream
  template <typename Z>
  friend std::ostream &operator<<(std::ostream&, const Matrix<Z>&);

  // Destructor
  V_DTR(~Matrix());
};

// Method implementation
#include "MatrixAccessors.ipp"
#include "MatrixConstructors.ipp"
#include "MatrixUtils.ipp"
#include "MatrixVirtuals.ipp"

namespace CoolDiff {
  namespace TensorR2 {
    namespace Details {

      // DevalR computation for matrix forward mode differentiation
      template <typename T>
      void DevalR(T& exp, const Matrix<Variable>& X, Matrix<Type>*& result) {
        const size_t nrows_x = X.getNumRows();
        const size_t ncols_x = X.getNumColumns();

        if (nullptr == result) {
          result = Matrix<Type>::MatrixFactory::CreateMatrixPtr(nrows_x, ncols_x);
        } else if ((nrows_x != result->getNumRows()) || (ncols_x != result->getNumColumns())) {
          result = Matrix<Type>::MatrixFactory::CreateMatrixPtr(nrows_x, ncols_x);
        }

        const size_t n_size = X.getNumElem();
        // Precompute (By design, the operation is serial)
        if constexpr (true == std::is_same_v<Expression, T>) {
          CoolDiff::TensorR1::PreComp(exp);
          std::transform(EXECUTION_SEQ X.getMatrixPtr(), X.getMatrixPtr() + n_size, result->getMatrixPtr(), 
                        [&exp](const auto &v) { return CoolDiff::TensorR1::DevalR(exp, v); });
        } else {
          // Create a new expression
          Expression exp2{exp};
          CoolDiff::TensorR1::PreComp(exp2);
          std::transform(EXECUTION_SEQ X.getMatrixPtr(), X.getMatrixPtr() + n_size, result->getMatrixPtr(),
                        [&exp2](const auto &v) { return CoolDiff::TensorR1::DevalR(exp2, v); });
        }
      }
      
      // Return scalar value for matrix based on it's type
      Type ScalarSpl(const Matrix<Type>*); 

      // Random number generation
      static std::random_device rd;
      static std::mt19937 gen(rd());

      // Fill matrix with random weights
      template<template <typename> class T, typename... Args>
      void FillRandomValues(MatType& M, Args&&... args) {
          T<Type> dis(std::forward<Args>(args)...);
          for(int i{}; i < M.getNumRows(); ++i) {
              for(int j{}; j < M.getNumColumns(); ++j) {
                  M(i,j) = dis(gen);
              }
          }
      }
    }
  }
}
