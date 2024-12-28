/**
 * @file include/Matrix/Matrix.hpp
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

#pragma once

#include "CommonFunctions.hpp"
#include "IMatrix.hpp"

// Derivative of matrices (Reverse AD)
Matrix<Type> *DervMatrix(const size_t, const size_t, const size_t,
                         const size_t);

// Matrix class
template <typename T> class Matrix : public IMatrix<Matrix<T>> {
public:
  // Matrix factory
  class MatrixFactory {
  public:
    // Create matrix as reference
    template <typename... Args> static Matrix<T> &CreateMatrix(Args &&...args) {
      auto tmp = Allocate<Matrix<T>>(std::forward<Args>(args)...);
      return *tmp;
    }

    // Create matrix as pointer
    template <typename... Args>
    static Matrix<T> *CreateMatrixPtr(Args &&...args) {
      auto tmp = Allocate<Matrix<T>>(std::forward<Args>(args)...);
      return tmp.get();
    }
  };

private:
  // Allocate friend function
  template <typename Z, typename... Argz>
  friend SharedPtr<Z> Allocate(Argz &&...);

  // Memory manager is a friend class
  friend class MemoryManager;

  // Matrices of template types is a friend class
  template <typename Z> friend class Matrix;

  // Special matrix constructor (Privatized, only for internal factory view)
  Matrix(const size_t, const size_t, const MatrixSpl &);

  // Matrix row and column size
  size_t m_rows{0};
  size_t m_cols{0};

  // Type of matrix (Special matrices)
  MatrixSpl m_type{(size_t)(-1)};

  // Matrix raw pointer of underlying type (Expression, Variable, Parameter,
  // Type)
  T *mp_mat{nullptr};

  // Collection of meta variable expressions
  Vector<MetaMatrix *> m_gh_vec{};

  // Free matrix resource
  bool m_free{false};

  // Boolean to verify evaluation/forward derivative values
  bool m_eval{false};
  bool m_devalf{false};

  // Matrix pointer for evaluation result (Type)
  Matrix<Type> *mp_result{nullptr};
  // Matrix pointer for forward derivative (Type)
  Matrix<Type> *mp_dresult{nullptr};

  // Set values for the result matrix
  void setEval();
  // Set value for the derivative result matrix
  void setDevalF(const Matrix<Variable> &);

  // Move constructor
  Matrix(Matrix &&) noexcept;
  // Move assignment operator
  Matrix &operator=(Matrix &&) noexcept;

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
  Matrix(const size_t, const size_t, const T &);

  // Matrix expressions constructor
  template <typename Z>
  Matrix(const IMatrix<Z> &expr)
      : m_rows{expr.getNumRows()}, m_cols{expr.getNumColumns()},
        m_type{(size_t)(-1)}, mp_mat{nullptr}, mp_result{nullptr},
        mp_dresult{nullptr}, m_eval{false}, m_devalf{false},
        m_nidx{this->m_idx_count++} {
    // Static assert so that type T is an expression
    static_assert(true == std::is_same_v<T, Expression>,
                  "[ERROR] The type T is not an expression");
    // Reserve a buffer of Matrix expressions
    m_gh_vec.reserve(g_vec_init);
    // Emplace the expression in a generic holder
    m_gh_vec.push_back((Matrix<Expression> *)&expr);
  }
  /* Copy assignment for expression evaluation */
  template <typename Z> Matrix &operator=(const IMatrix<Z> &expr) {
    // Static assert so that type T is an expression
    static_assert(true == std::is_same_v<T, Expression>,
                  "[ERROR] The type T is not an expression");
    // Clear buffer and set rows and columns if not recursive expression not
    // found
    if (static_cast<const Z &>(expr).findMe(this) == false) {
      m_gh_vec.clear();
    }
    // If the push back bector is zero
    if (true == m_gh_vec.empty()) {
      m_rows = expr.getNumRows();
      m_cols = expr.getNumColumns();
      m_type = SIZE_MAX;
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
    }
    // Emplace the expression in a generic holder
    m_gh_vec.push_back((Matrix<Expression> *)&expr);
    return *this;
  }

  // Copy constructor
  Matrix(const Matrix &);
  // Copy assignment operator
  Matrix &operator=(const Matrix &);

  // Get matrix pointer immutable
  const T *getMatrixPtr() const;
  // Get matrix pointer mutable
  T *getMatrixPtr();

  // Matrix 2D access using operator()() immutable
  const T &operator()(const size_t, const size_t) const;
  // Matrix 2D access using operator()() mutable
  T &operator()(const size_t, const size_t);

  // Matrix 1D access using operator[] immutable
  const T &operator[](const size_t) const;
  // Matrix 1D access using operator[] mutable
  T &operator[](const size_t);

  // Get block matrix
  void getBlockMat(const Pair<size_t, size_t> &rows,
                   const Pair<size_t, size_t> &cols, Matrix *&result) const {
    const size_t row_start = rows.first;
    const size_t row_end = rows.second;
    const size_t col_start = cols.first;
    const size_t col_end = cols.second;

    // Assert for row start/end, column start/end and index out of bound checks
    ASSERT((row_start >= 0 && row_start < m_rows),
           "Row starting index out of bound");
    ASSERT((row_end >= 0 && row_end < m_rows), "Row ending index out of bound");
    ASSERT((col_start >= 0 && col_start < m_cols),
           "Column starting index out of bound");
    ASSERT((col_end >= 0 && col_end < m_cols),
           "Column ending index out of bound");
    ASSERT((row_start <= row_end), "Row start greater than row ending");
    ASSERT((col_start <= col_end), "Column start greater than row ending");

    // TODO Special matrix embedding
    if (getMatType() == MatrixSpl::ZEROS) {
      result = MemoryManager::MatrixSplPool(
          row_end - row_start + 1, col_end - col_start + 1, MatrixSpl::ZEROS);
    } else {
      MemoryManager::MatrixPool(row_end - row_start + 1,
                                col_end - col_start + 1, result);
      const auto outer_idx = Range<size_t>(row_start, row_end + 1);
      const auto inner_idx = Range<size_t>(col_start, col_end + 1);
      std::for_each(
          EXECUTION_PAR outer_idx.begin(), outer_idx.end(),
          [this, &col_start, &row_start, &inner_idx, result](const size_t i) {
            std::for_each(EXECUTION_PAR inner_idx.begin(), inner_idx.end(),
                          [i, this, &col_start, &row_start, &inner_idx,
                           result](const size_t j) {
                            (*result)(i - row_start, j - col_start) =
                                (*this)(i, j);
                          });
          });
    }
  }

  // Set block matrix
  void setBlockMat(const Pair<size_t, size_t> &rows,
                   const Pair<size_t, size_t> &cols, const Matrix *result) {
    const size_t row_start = rows.first;
    const size_t row_end = rows.second;
    const size_t col_start = cols.first;
    const size_t col_end = cols.second;

    // Assert for row start/end, column start/end and index out of bound checks
    ASSERT((row_start >= 0 && row_start < m_rows),
           "Row starting index out of bound");
    ASSERT((row_end >= 0 && row_end < m_rows), "Row ending index out of bound");
    ASSERT((col_start >= 0 && col_start < m_cols),
           "Column starting index out of bound");
    ASSERT((col_end >= 0 && col_end < m_cols),
           "Column ending index out of bound");
    ASSERT((row_start <= row_end), "Row start greater than row ending");
    ASSERT((col_start <= col_end), "Column start greater than row ending");
    ASSERT((row_end - row_start + 1 == result->getNumRows()),
           "Row mismatch for insertion matrix");
    ASSERT((col_end - col_start + 1 == result->getNumColumns()),
           "Column mismatch for insertion matrix");

    // Special matrix embedding
    const auto outer_idx = Range<size_t>(row_start, row_end + 1);
    const auto inner_idx = Range<size_t>(col_start, col_end + 1);

    // TODO Special matrix embedding
    if (result->getMatType() == MatrixSpl::ZEROS) {
      if constexpr (std::is_same_v<T, Type>) {
        std::for_each(
            EXECUTION_PAR outer_idx.begin(), outer_idx.end(),
            [this, &col_start, &row_start, &inner_idx, result](const size_t i) {
              std::for_each(
                  EXECUTION_PAR inner_idx.begin(), inner_idx.end(),
                  [i, this, &col_start, &row_start, &inner_idx,
                   result](const size_t j) { (*this)(i, j) = (Type)(0); });
            });
      }
    } else {
      std::for_each(
          EXECUTION_PAR outer_idx.begin(), outer_idx.end(),
          [this, &col_start, &row_start, &inner_idx, result](const size_t i) {
            std::for_each(EXECUTION_PAR inner_idx.begin(), inner_idx.end(),
                          [i, this, &col_start, &row_start, &inner_idx,
                           result](const size_t j) {
                            (*this)(i, j) =
                                (*result)(i - row_start, j - col_start);
                          });
          });
    }
  }

  // Add zero padding
  void pad(const size_t r, const size_t c, Matrix *&result) const {
    // Special matrix embedding
    MemoryManager::MatrixPool(m_rows + 2 * r, m_cols + 2 * c, result);
    result->setBlockMat({r, r + m_rows - 1}, {c, c + m_cols - 1}, this);
  }

  // Get a row for matrix using move semantics
  Matrix getRow(const size_t) &&;
  // Get a row for matrix using copy semantics
  Matrix getRow(const size_t) const &;
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

  // Get type of matrix
  MatrixSpl getMatType() const;

  // Free resources
  void free();
  // Find me
  bool findMe(void *) const;
  // Reset impl
  void resetImpl();

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const);
  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const);

  // Evaluate matrix
  V_OVERRIDE(Matrix<Type> *eval());
  // Derivative matrix
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &));
  // Reset all visited flags
  V_OVERRIDE(void reset());

  // Get type
  V_OVERRIDE(std::string_view getType() const);

  // To output stream
  template <typename Z>
  friend std::ostream &operator<<(std::ostream &, Matrix<Z> &);

  // Destructor
  V_DTR(~Matrix());
};

#include "MatrixAccessors.ipp"
#include "MatrixConstructors.ipp"
#include "MatrixUtils.ipp"
#include "MatrixVirtuals.ipp"