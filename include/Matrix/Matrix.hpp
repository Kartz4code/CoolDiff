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

// Factory function for matrix reference creation
template <typename T> Matrix<T> &CreateMatrix(size_t, size_t);

// Factory function for matrix pointer creation
template <typename T> Matrix<T> *CreateMatrixPtr(size_t, size_t);

// Matrix class
template <typename T> class Matrix : public IMatrix<Matrix<T>> {
private:
  // Matrix row and column size
  size_t m_rows{0};
  size_t m_cols{0};

  // Collection of meta variable expressions
  Vector<MetaMatrix *> m_gh_vec{};

  // Get value
  inline constexpr Type getValue(T &val) const {
    // If T is of type Type
    if constexpr (true == std::is_same_v<T, Type>) {
      return val;
      // If T is of type Variable
    } else if constexpr (true == std::is_same_v<T, Variable> ||
                         true == std::is_same_v<T, Parameter>) {
      return val.eval();
      // If T is of type Expression
    } else if constexpr (true == std::is_same_v<T, Expression>) {
      return Eval(val);
    } else {
      // If T is unknown, then return typecasted val
      return (Type)(val);
    }
  }

  // Get derivative value
  inline constexpr Type getdValue(T &val, const Variable &var) const {
    // If T is of type Type
    if constexpr (true == std::is_same_v<T, Type>) {
      return (Type)(0);
      // If T is of type Variable
    } else if constexpr (true == std::is_same_v<T, Variable> ||
                         true == std::is_same_v<T, Parameter>) {
      return val.devalF(var);
      // If T is of type Expression
    } else if constexpr (true == std::is_same_v<T, Expression>) {
      return DevalF(val, var);
    } else {
      // If T is unknown, then return typecasted 0
      return (Type)(0);
    }
  }

public:
  // Matrix raw pointer of underlying type (Expression, Variable, Parameter,
  // Type)
  T *mp_mat{nullptr};
  // Matrix pointer for evaluation result (Type)
  Matrix<Type> *mp_result{nullptr};
  // Matrix pointer for forward derivative (Type)
  Matrix<Type> *mp_dresult{nullptr};

  // Block index
  size_t m_nidx{};

  // Default constructor
  Matrix()
      : m_rows{0}, m_cols{0}, mp_mat{new T{}}, mp_result{nullptr},
        mp_dresult{nullptr}, m_nidx{this->m_idx_count++} {}

  // Initalize the matrix with rows and columns
  Matrix(size_t rows, size_t cols)
      : m_rows{rows}, m_cols{cols}, mp_mat{new T[getNumElem()]{}},
        mp_result{nullptr}, mp_dresult{nullptr}, m_nidx{this->m_idx_count++} {}

  // Matrix expressions constructor
  template <typename Z>
  Matrix(const IMatrix<Z> &expr)
      : m_rows{expr.getNumRows()}, m_cols{expr.getNumColumns()},
        mp_mat{new T[getNumElem()]{}}, mp_result{nullptr},
        mp_dresult{nullptr}, m_nidx{this->m_idx_count++} {
    // Reserve a buffer of Matrix expressions
    m_gh_vec.reserve(g_vec_init);
    // Emplace the expression in a generic holder
    m_gh_vec.emplace_back(&static_cast<const Z &>(expr));
  }

  /* Copy assignment for expression evaluation */
  template <typename Z> Matrix &operator=(const IMatrix<Z> &expr) {
    if (static_cast<const Z &>(expr).findMe(this) == false) {
      m_gh_vec.clear();
    }
    // Emplace the expression in a generic holder
    m_gh_vec.emplace_back(&static_cast<const Z &>(expr));
    return *this;
  }

  // Move constructor
  Matrix(Matrix &&m) noexcept
      : m_rows{std::exchange(m.m_rows, -1)}, m_cols{std::exchange(m.m_cols,
                                                                  -1)},
        mp_mat{std::exchange(m.mp_mat, nullptr)}, mp_result{std::exchange(
                                                      m.mp_result, nullptr)},
        mp_dresult{std::exchange(m.mp_dresult, nullptr)},
        m_gh_vec{std::exchange(m.m_gh_vec, {})}, m_nidx{std::exchange(m.m_nidx,
                                                                      -1)} {}

  // Move assignment operator
  Matrix &operator=(Matrix &&m) noexcept {
    if (nullptr != mp_mat) {
      delete[] mp_mat;
    }

    // Exchange values
    mp_mat = std::exchange(m.mp_mat, nullptr);
    m_rows = std::exchange(m.m_rows, -1);
    m_cols = std::exchange(m.m_cols, -1);
    mp_result = std::exchange(m.mp_result, nullptr);
    mp_dresult = std::exchange(m.mp_dresult, nullptr);
    m_nidx = std::exchange(m.m_nidx, -1);
    m_gh_vec = std::exchange(m.m_gh_vec, {});

    // Return this reference
    return *this;
  }

  // Copy constructor
  Matrix(const Matrix &m)
      : m_rows{m.m_rows}, m_cols{m.m_cols}, mp_mat{new T[getNumElem()]{}},
        m_nidx{m.m_nidx}, m_gh_vec{m.m_gh_vec} {
    // Copy values
    std::copy(EXECUTION_PAR m.mp_mat, m.mp_mat + getNumElem(), mp_mat);

    // Clone mp_result
    if (nullptr != m.mp_result) {
      mp_result = m.mp_result->clone();
    }
    // Clone mp_dresult
    if (nullptr != m.mp_dresult) {
      mp_dresult = m.mp_dresult->clone();
    }
  }

  // Copy assignment operator
  Matrix &operator=(const Matrix &m) {
    if (&m != this) {
      // Assign resources
      m_rows = m.m_rows;
      m_cols = m.m_cols;
      m_nidx = m.m_nidx;
      m_gh_vec = m.m_gh_vec;

      // Copy mp_mat
      if (nullptr != mp_mat) {
        delete[] mp_mat;
      }
      mp_mat = new T[getNumElem()]{};
      std::copy(EXECUTION_PAR m.mp_mat, m.mp_mat + getNumElem(), mp_mat);

      // Copy and clone mp_result
      if (nullptr != mp_result) {
        delete[] mp_result;
      }
      if (m.mp_result != nullptr) {
        mp_result = m.mp_result->clone();
      }

      // Copy and clone mp_dresult
      if (nullptr != mp_dresult) {
        delete[] mp_dresult;
      }
      if (m.mp_dresult != nullptr) {
        mp_dresult = m.mp_dresult->clone();
      }
    }

    // Return this reference
    return *this;
  }

  // For cloning numerical values for copy constructors
  inline Matrix<Type> *clone() const {
    Matrix<Type> *result = CreateMatrixPtr<Type>(m_rows, m_cols);
    if (nullptr != result->mp_mat) {
      std::copy(EXECUTION_PAR mp_mat, mp_mat + getNumElem(), result->mp_mat);
    }
    return result;
  }

  // Reshape matrix (rows and columns)
  void reshape(size_t rows, size_t cols) {
    assert((rows * cols == getNumElem()) &&
           "[ERROR] Matrix resize operation invalid");
    // Assign new rows and cols
    m_rows = rows;
    m_cols = cols;
  }

  // Get matrix pointer const
  const T *getMatrixPtr() const { return mp_mat; }

  // Get matrix pointer
  T *getMatrixPtr() { return mp_mat; }

  // Matrix 2D access using operator()()
  T &operator()(const size_t i, const size_t j) {
    assert((i >= 0 && i < m_rows) && "[ERROR] Row index out of bound");
    assert((j >= 0 && j < m_cols) && "[ERROR] Column index out of bound");
    return mp_mat[i * m_cols + j];
  }

  // Matrix 2D access using operator()() const
  const T &operator()(const size_t i, const size_t j) const {
    assert((i >= 0 && i < m_rows) && "[ERROR] Row index out of bound");
    assert((j >= 0 && j < m_cols) && "[ERROR] Column index out of bound");
    return mp_mat[i * m_cols + j];
  }

  // Matrix 1D access using operator[]
  T &operator[](const size_t l) {
    assert((l >= 0 && l < getNumElem()) && "[ERROR] Index out of bound");
    return mp_mat[l];
  }

  // Matrix 1D access using operator[] const
  const T &operator[](const size_t l) const {
    assert((l >= 0 && l < getNumElem()) && "[ERROR] Index out of bound");
    return mp_mat[l];
  }

  // Get a row (Move)
  Matrix getRow(const size_t &i) && {
    assert((i >= 0 && i < m_rows) && "[ERROR] Row index out of bound");
    Matrix tmp(m_cols, 1);
    std::copy(EXECUTION_PAR mp_mat + (i * m_cols),
              mp_mat + ((i + 1) * m_cols), tmp.getMatrixPtr());
    return std::move(tmp);
  }

  // Get a row (Copy)
  Matrix getRow(const size_t &i) const & {
    assert((i >= 0 && i < m_rows) && "[ERROR] Row index out of bound");
    Matrix tmp(m_cols, 1);
    std::copy(EXECUTION_PAR mp_mat + (i * m_cols),
              mp_mat + ((i + 1) * m_cols), tmp.getMatrixPtr());
    return tmp;
  }

  // Get a column (Move)
  Matrix getColumn(const size_t &i) && {
    assert((i >= 0 && i < m_cols) && "[ERROR] Column index out of bound");
    Matrix tmp(m_rows, 1);
    for (size_t j{}; j < m_rows; ++j) {
      tmp.mp_mat[j] = mp_mat[j * m_rows + i];
    }
    return std::move(tmp);
  }

  // Get a column (copy)
  Matrix getColumn(const size_t &i) const & {
    assert((i >= 0 && i < m_cols) && "[ERROR] Column index out of bound");
    Matrix tmp(m_rows, 1);
    for (size_t j{}; j < m_rows; ++j) {
      tmp.mp_mat[j] = mp_mat[j * m_rows + i];
    }
    return tmp;
  }

  // Get total elements
  size_t getNumElem() const { return (m_rows * m_cols); }

  // Get number of rows
  size_t getNumRows() const { return m_rows; }

  // Get number of columns
  size_t getNumColumns() const { return m_cols; }

  // Find me
  bool findMe(void *v) const {
    if (static_cast<const void *>(this) == v) {
      return true;
    } else {
      return false;
    }
  }

  V_OVERRIDE(Matrix<Type> *eval()) {
    // Cache the mp_result value
    if (false == this->m_visited) {
      if (nullptr == mp_result) {
        // [Optimization - Scope for Type type]
        mp_result = CreateMatrixPtr<Type>(m_rows, m_cols);
      }

      // Set result matrix
      std::transform(EXECUTION_PAR mp_mat, mp_mat + getNumElem(),
                     mp_result->mp_mat,
                     [this](auto &v) { return getValue(v); });

      // Set visit flag to true
      this->m_visited = true;
      // Loop on internal equations
      for (auto &i : m_gh_vec) {
        if (nullptr != i) {
          mp_result = i->eval();
        }
      }
    }

    // Return evaulation result
    return mp_result;
  }

  V_OVERRIDE(Matrix<Type> *devalF(const Variable &var)) {
    // Cache the mp_result value
    if (false == this->m_visited) {
      // Derivative result computation
      if (nullptr == mp_dresult) {
        mp_dresult = CreateMatrixPtr<Type>(m_rows, m_cols);
      }

      // Set derivative result matrix
      std::transform(EXECUTION_PAR mp_mat, mp_mat + getNumElem(),
                     mp_dresult->mp_mat,
                     [&var, this](auto &v) { return getdValue(v, var); });

      // Set visit flag to true
      this->m_visited = true;
      // Loop on internal equations
      for (auto &i : m_gh_vec) {
        if (nullptr != i) {
          mp_dresult = i->devalF(var);
          mp_result = i->eval();
        }
      }
    }

    // Return derivative result
    return mp_dresult;
  }

  // Reset all visited flags
  V_OVERRIDE(void reset()) {
    if (true == this->m_visited) {
      this->m_visited = false;
      for (auto &i : m_gh_vec) {
        if (nullptr != i) {
          i->reset();
        }
      }
    }
    // Reset flag
    this->m_visited = false;
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { return "Matrix"; }

  // Reset impl
  inline void resetImpl() {
    this->m_visited = true;
    // Reset states
    for (auto &i : m_gh_vec) {
      if (i != nullptr) {
        i->reset();
      }
    }
    this->m_visited = false;
  }

  // To output stream
  friend std::ostream &operator<<(std::ostream &os, Matrix &mat) {
    for (size_t i{}; i < mat.getNumElem(); ++i) {
      os << mat.getValue(mat.mp_mat[i]) << " ";
    }
    os << "\n";
    return os;
  }

  // Destructor
  V_DTR(~Matrix()) {
    // If mp_mat is not nullptr, delete it
    if (nullptr != mp_mat) {
      delete[] mp_mat;
      mp_mat = nullptr;
    }
  }
};

// Factory function for matrix creation
template <typename T> Matrix<T> &CreateMatrix(size_t rows, size_t cols) {
  auto tmp = Allocate<Matrix<T>>(rows, cols);
  return *tmp;
}

template <typename T> Matrix<T> *CreateMatrixPtr(size_t rows, size_t cols) {
  auto tmp = Allocate<Matrix<T>>(rows, cols);
  return tmp.get();
}