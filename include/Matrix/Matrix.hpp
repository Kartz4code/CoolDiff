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


// Get value
template<typename T>
inline constexpr Type GetValue(T &val) {
  // If T is of type Type
  if constexpr (true == std::is_same_v<T, Type>) {
    return val;

  }
  // If T is of type Parameter
  else if constexpr (true == std::is_same_v<T, Parameter>) {
    return val.eval();
  }
  // If T is of type Variable
  else if constexpr (true == std::is_same_v<T, Variable>) {
    val.resetImpl();
    return val.eval();
  }
  // If T is of type Expression
  else if constexpr (true == std::is_same_v<T, Expression>) {
    return Eval(val);
  } else {
    // If T is unknown, then return typecasted val
    return (Type)(val);
  }
}

// Get derivative value (Forward derivative)
template<typename T>
inline constexpr Type GetDValue(T &val, const Variable &var) {
  // If T is of type Variable
  if constexpr (true == std::is_same_v<T, Variable>) {
    val.resetImpl();
    return val.devalF(var);
    // If T is of type Expression
  } else if constexpr (true == std::is_same_v<T, Expression>) {
    return DevalF(val, var);
  } else {
    // If T is unknown, then return typecasted 0
    return (Type)(0);
  }
}

// Factory function for matrix reference creation
template <typename T, typename... Args> Matrix<T> &CreateMatrix(Args &&...);

// Factory function for matrix pointer creation
template <typename T, typename... Args> Matrix<T> *CreateMatrixPtr(Args &&...);

// Matrix class
template <typename T> class Matrix : public IMatrix<Matrix<T>> {
private:
  // Matrix row and column size
  mutable size_t m_rows{0};
  mutable size_t m_cols{0};

  // Type of matrix (Special matrices)
  size_t m_type{(size_t)(-1)};

  // Collection of meta variable expressions
  Vector<MetaMatrix *> m_gh_vec{};

  // Set values for the result matrix
  inline void setEval() {
    // Set result matrix
    if constexpr (false == std::is_same_v<T, Type>) {
      if ((nullptr != mp_mat) && (nullptr != mp_result) &&
          (nullptr != mp_result->mp_mat)) {
        std::transform(EXECUTION_SEQ mp_mat, mp_mat + getNumElem(),
                       mp_result->mp_mat,
                       [this](auto &v) { return GetValue(v); });
      }
    }
  }

  // Set value for the derivative result matrix (Matrix)
  inline void setDevalF(const Matrix<Variable> &X) {
    // If the matrix type is Expression
    if constexpr (true == std::is_same_v<T, Expression>) {
      if ((nullptr != mp_mat) && (nullptr != mp_dresult) &&
          (nullptr != mp_dresult->mp_mat)) {
        // Precompute the reverse derivatives
        std::for_each(EXECUTION_SEQ mp_mat, mp_mat + getNumElem(),
                      [](auto &i) { PreComp(i); });

        // Get dimensions of X variable matrix
        const size_t xrows = X.getNumRows();
        const size_t xcols = X.getNumColumns();

        // Vector of indices in current matrix
        auto outer_idx = Range<size_t>(0, getNumElem());
        auto inner_idx = Range<size_t>(0, xrows * xcols);

        // Logic for Kronecker product (Reverse mode differentiation)
        std::for_each(EXECUTION_PAR outer_idx.begin(), outer_idx.end(),
                      [this, &X, &inner_idx, xrows, xcols](const size_t i) {
                        const size_t k = i % m_cols;
                        const size_t l = (i - k) / m_cols;

                        // Inner loop
                        std::for_each(EXECUTION_PAR inner_idx.begin(),
                                      inner_idx.end(), [&](const size_t n) {
                                        const size_t j = n % xcols;
                                        const size_t i = (n - j) / xcols;
                                        (*mp_dresult)(l * xrows + i,
                                                      k * xcols + j) =
                                            DevalR((*this)(l, k), X(i, j));
                                      });
                      });
      }
    }
    // If the matrix type is Variable
    else if constexpr (true == std::is_same_v<T, Variable>) {
      if ((nullptr != mp_mat) && (nullptr != mp_dresult) &&
          (nullptr != mp_dresult->mp_mat) && (m_nidx == X.m_nidx)) {
        // Get dimensions of X variable matrix
        const size_t xrows = X.getNumRows();
        const size_t xcols = X.getNumColumns();

        // Vector of indices in X matrix
        auto idx = Range<size_t>(0, X.getNumElem());
        // Logic for Kronecker product (With ones)
        std::for_each(EXECUTION_PAR idx.begin(), idx.end(),
                      [this, xrows, xcols](const size_t n) {
                        const size_t j = n % xcols;
                        const size_t i = (n - j) / xcols;
                        // Inner loop
                        (*mp_dresult)(i * xrows + i, j * xcols + j) = (Type)(1);
                      });
      }
      // If the Matrix is Variable, but of different form
      else if ((nullptr != mp_mat) && (nullptr != mp_dresult) &&
               (nullptr != mp_dresult->mp_mat)) {
        // Get dimensions of X variable matrix
        const size_t xrows = X.getNumRows();
        const size_t xcols = X.getNumColumns();

        // Vector of indices in current matrix
        auto outer_idx = Range<size_t>(0, getNumElem());
        auto inner_idx = Range<size_t>(0, xrows * xcols);

        // Logic for Kronecker product (Forward mode differentiation)
        std::for_each(EXECUTION_PAR outer_idx.begin(), outer_idx.end(),
                      [this, &X, &inner_idx, xrows, xcols](const size_t i) {
                        const size_t k = i % m_cols;
                        const size_t l = (i - k) / m_cols;

                        // Inner loop
                        std::for_each(
                            EXECUTION_PAR inner_idx.begin(), inner_idx.end(),
                            [&](const size_t n) {
                              const size_t j = n % xcols;
                              const size_t i = (n - j) / xcols;
                              (*mp_dresult)(l * xrows + i, k * xcols + j) =
                                  (((*this)(l, k).m_nidx == X(i, j).m_nidx)
                                       ? (Type)(1)
                                       : (Type)(0));
                            });
                      });
      }
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
  // Boolean to verify evaluation/forward derivative values
  bool m_eval{false}, m_devalf{false};
  // Block index
  size_t m_nidx{};

  // Special matrix constructor
  Matrix(size_t rows, size_t cols, size_t type)
      : m_rows{rows}, m_cols{cols}, m_type{type} {}

  // Default constructor
  Matrix()
      : m_rows{0}, m_cols{0}, m_type{(size_t)(-1)}, mp_mat{new T{}},
        mp_result{nullptr}, mp_dresult{nullptr}, m_eval{false}, m_devalf{false},
        m_nidx{this->m_idx_count++} {}

  // Initalize the matrix with rows and columns
  Matrix(size_t rows, size_t cols)
      : m_rows{rows}, m_cols{cols}, m_type{(size_t)(-1)},
        mp_mat{new T[getNumElem()]{}}, mp_result{nullptr}, mp_dresult{nullptr},
        m_eval{false}, m_devalf{false}, m_nidx{this->m_idx_count++} {}

  // Matrix expressions constructor
  template <typename Z>
  Matrix(const IMatrix<Z> &expr)
      : m_rows{expr.getNumRows()}, m_cols{expr.getNumColumns()},
        m_type{(size_t)(-1)}, mp_result{nullptr}, mp_dresult{nullptr},
        m_eval{false}, m_devalf{false}, m_nidx{this->m_idx_count++} {
    // Reserve a buffer of Matrix expressions
    m_gh_vec.reserve(g_vec_init);
    // Emplace the expression in a generic holder
    m_gh_vec.emplace_back(&static_cast<const Z &>(expr));
  }

  /* Copy assignment for expression evaluation */
  template <typename Z> Matrix &operator=(const IMatrix<Z> &expr) {
    // Clear buffer if not recursive expression not found
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
        m_type{std::exchange(m.m_type, -1)}, mp_mat{std::exchange(m.mp_mat,
                                                                  nullptr)},
        mp_result{std::exchange(m.mp_result, nullptr)},
        mp_dresult{std::exchange(m.mp_dresult, nullptr)},
        m_eval{std::exchange(m.m_eval, false)}, m_devalf{std::exchange(
                                                    m.m_devalf, false)},
        m_gh_vec{std::exchange(m.m_gh_vec, {})}, m_nidx{std::exchange(m.m_nidx,
                                                                      -1)}

  {}

  // Move assignment operator
  Matrix &operator=(Matrix &&m) noexcept {
    if (nullptr != mp_mat) {
      delete[] mp_mat;
    }

    // Exchange values
    mp_mat = std::exchange(m.mp_mat, nullptr);
    m_rows = std::exchange(m.m_rows, -1);
    m_cols = std::exchange(m.m_cols, -1);
    m_type = std::exchange(m.m_type, -1);
    mp_result = std::exchange(m.mp_result, nullptr);
    mp_dresult = std::exchange(m.mp_dresult, nullptr);
    m_eval = std::exchange(m.m_eval, false);
    m_devalf = std::exchange(m.m_devalf, false);
    m_nidx = std::exchange(m.m_nidx, -1);
    m_gh_vec = std::exchange(m.m_gh_vec, {});

    // Return this reference
    return *this;
  }

  // Copy constructor
  Matrix(const Matrix &m)
      : m_rows{m.m_rows}, m_cols{m.m_cols}, m_type{m.m_type},
        mp_mat{new T[getNumElem()]{}}, m_eval{m.m_eval}, m_devalf{m.m_devalf},
        m_nidx{m.m_nidx}, m_gh_vec{m.m_gh_vec} {
    // Copy values
    std::copy(EXECUTION_PAR m.mp_mat, m.mp_mat + getNumElem(), mp_mat);
  }

  // Copy assignment operator
  Matrix &operator=(const Matrix &m) {
    if (&m != this) {
      // Assign resources
      m_rows = m.m_rows;
      m_cols = m.m_cols;
      m_type = m.m_type;
      m_nidx = m.m_nidx;
      m_gh_vec = m.m_gh_vec;
      m_eval = m.m_eval;
      m_devalf = m.m_devalf;

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
      // Copy and clone mp_dresult
      if (nullptr != mp_dresult) {
        delete[] mp_dresult;
      }
    }

    // Return this reference
    return *this;
  }

  // Get matrix pointer const
  const T *getMatrixPtr() const { return mp_mat; }

  // Get matrix pointer
  T *getMatrixPtr() { return mp_mat; }

  // Matrix 2D access using operator()()
  T &operator()(const size_t i, const size_t j) {
    ASSERT((i >= 0 && i < m_rows), "Row index out of bound");
    ASSERT((j >= 0 && j < m_cols), "Column index out of bound");
    return mp_mat[i * m_cols + j];
  }

  // Matrix 2D access using operator()() const
  const T &operator()(const size_t i, const size_t j) const {
    ASSERT((i >= 0 && i < m_rows), "Row index out of bound");
    ASSERT((j >= 0 && j < m_cols), "Column index out of bound");
    return mp_mat[i * m_cols + j];
  }

  // Matrix 1D access using operator[]
  T &operator[](const size_t l) {
    ASSERT((l >= 0 && l < getNumElem()), "Index out of bound");
    return mp_mat[l];
  }

  // Matrix 1D access using operator[] const
  const T &operator[](const size_t l) const {
    ASSERT((l >= 0 && l < getNumElem()), "Index out of bound");
    return mp_mat[l];
  }

  // Get a row (Move)
  Matrix getRow(const size_t &i) && {
    ASSERT((i >= 0 && i < m_rows), "Row index out of bound");
    Matrix tmp(m_cols, 1);
    std::copy(EXECUTION_PAR mp_mat + (i * m_cols), mp_mat + ((i + 1) * m_cols),
              tmp.getMatrixPtr());
    return std::move(tmp);
  }

  // Get a row (Copy)
  Matrix getRow(const size_t &i) const & {
    ASSERT((i >= 0 && i < m_rows), "Row index out of bound");
    Matrix tmp(m_cols, 1);
    std::copy(EXECUTION_PAR mp_mat + (i * m_cols), mp_mat + ((i + 1) * m_cols),
              tmp.getMatrixPtr());
    return tmp;
  }

  // Get a column (Move)
  Matrix getColumn(const size_t &i) && {
    ASSERT((i >= 0 && i < m_cols), "Column index out of bound");
    Matrix tmp(m_rows, 1);

    // Iteration elements
    auto idx = Range<size_t>(0, m_rows);
    // For each execution
    std::for_each(EXECUTION_PAR idx.begin(), idx.end(),
                  [this, &tmp](const size_t n) {
                    const size_t j = n % m_cols;
                    const size_t i = (n - j) / m_cols;
                    tmp.mp_mat[j] = mp_mat[j * m_rows + i];
                  });

    return std::move(tmp);
  }

  // Get a column (copy)
  Matrix getColumn(const size_t &i) const & {
    ASSERT((i >= 0 && i < m_cols), "Column index out of bound");
    Matrix tmp(m_rows, 1);

    // Iteration elements
    auto idx = Range<size_t>(0, m_rows);
    // For each execution
    std::for_each(EXECUTION_PAR idx.begin(), idx.end(),
                  [this, &tmp](const size_t n) {
                    const size_t j = n % m_cols;
                    const size_t i = (n - j) / m_cols;
                    tmp.mp_mat[j] = mp_mat[j * m_rows + i];
                  });

    return tmp;
  }

  // Get number of rows
  V_OVERRIDE(size_t getNumRows() const) { return m_rows; }

  // Get number of columns
  V_OVERRIDE(size_t getNumColumns() const) { return m_cols; }

  // Get total elements
  size_t getNumElem() const { return (m_rows * m_cols); }

  // Get final number of rows (Expression)
  size_t getFinalNumRows() const {
    size_t rows{};
    if constexpr (std::is_same_v<T, Expression>) {
      if (false == m_gh_vec.empty()) {
        if (auto it = m_gh_vec.back(); nullptr != it) {
          rows = it->getNumRows();
        }
      }
    } else {
      rows = getNumRows();
    }
    return rows;
  }

  // Get final number of columns (Expression)
  size_t getFinalNumColumns() const {
    size_t cols{};
    if constexpr (std::is_same_v<T, Expression>) {
      if (false == m_gh_vec.empty()) {
        if (auto it = m_gh_vec.back(); nullptr != it) {
          cols = it->getNumColumns();
        }
      }
    } else {
      cols = getNumColumns();
    }
    return cols;
  }

  // Get total final number of elements
  size_t getFinalNumElem() const {
    return getFinalNumRows() * getFinalNumColumns();
  }

  // Get type of matrix
  size_t getMatType() const { return m_type; }

  // Find me
  bool findMe(void *v) const {
    if (static_cast<const void *>(this) == v) {
      return true;
    } else {
      return false;
    }
  }

  // Evaluate matrix
  V_OVERRIDE(Matrix<Type> *eval()) {
    // Cache the mp_result value
    if (nullptr == mp_result) {
      if constexpr (false == std::is_same_v<T, Type>) {
        mp_result = CreateMatrixPtr<Type>(m_rows, m_cols);
      } else {
        mp_result = this;
      }
    }

    // If value not evaluated, compute it again
    if (false == m_eval) {
      setEval();
      m_eval = true;
    }

    // If visited already
    if (false == this->m_visited) {
      // Set visit flag to true
      this->m_visited = true;
      // Loop on internal equations
      std::for_each(EXECUTION_SEQ m_gh_vec.begin(), m_gh_vec.end(),
                    [this](auto *i) {
                      if (nullptr != i) {
                        mp_result = i->eval();
                        m_eval = true;
                      }
                    });
    }

    // Return evaulation result
    return mp_result;
  }

  // Derivative matrix
  V_OVERRIDE(Matrix<Type> *devalF(Matrix<Variable> &X)) {
    // Derivative result computation
    if (nullptr == mp_dresult) {
      const size_t xrows = X.getNumRows();
      const size_t xcols = X.getNumColumns();
      if constexpr (true == std::is_same_v<T, Type> ||
                    true == std::is_same_v<T, Parameter>) {
#if defined(NAIVE_IMPL)
        mp_dresult = CreateMatrixPtr<Type>(m_rows * xrows, m_cols * xcols,
                                           MatrixSpl::ZEROS);
#else
        mp_dresult = CreateMatrixPtr<Type>(m_rows * xrows, m_cols * xcols);
#endif
      } else {
        mp_dresult = CreateMatrixPtr<Type>(m_rows * xrows, m_cols * xcols);
      }
    }

    // If derivative not evaluated, compute it again
    if (false == m_devalf) {
      setDevalF(X);
      m_devalf = true;
    }

    // If visited already
    if (false == this->m_visited) {
      // Set visit flag to true
      this->m_visited = true;
      // Loop on internal equations
      std::for_each(EXECUTION_SEQ m_gh_vec.begin(), m_gh_vec.end(),
                    [this, &X](auto *i) {
                      if (nullptr != i) {
                        mp_dresult = i->devalF(X);
                        m_devalf = true;
                        mp_result = i->eval();
                        m_eval = true;
                      }
                    });
    }

    // Return derivative result
    return mp_dresult;
  }

  // Reset all visited flags
  V_OVERRIDE(void reset()) {
    if (true == this->m_visited) {
      this->m_visited = false;
      // Reset states
      m_eval = false;
      m_devalf = false;

      // For each element
      std::for_each(EXECUTION_SEQ m_gh_vec.begin(), m_gh_vec.end(),
                    [](auto *item) {
                      if (nullptr != item) {
                        item->reset();
                      }
                    });
    }
    // Reset flag
    this->m_visited = false;
  }

  // Reset impl
  inline void resetImpl() {
    this->m_visited = true;
    // Reset states
    m_eval = false;
    m_devalf = false;

    // For each element
    std::for_each(EXECUTION_SEQ m_gh_vec.begin(), m_gh_vec.end(),
                  [](auto *item) {
                    if (nullptr != item) {
                      item->reset();
                    }
                  });

    this->m_visited = false;
  }

  // Get type
  V_OVERRIDE(std::string_view getType() const) { return "Matrix"; }

  // To output stream
  friend std::ostream &operator<<(std::ostream &os, Matrix &mat) {
    if constexpr (true == std::is_same_v<T, Type>) {
      const size_t rows = mat.getFinalNumRows();
      const size_t cols = mat.getFinalNumColumns();
      if (mat.getMatType() == MatrixSpl::ZEROS) {
        os << "Zero matrix of dimension: "
           << "(" << rows << "," << cols << ")\n";
      } else if (mat.getMatType() == MatrixSpl::EYE) {
        os << "Identity matrix of dimension: "
           << "(" << rows << "," << cols << ")\n";
      } else {
        // Serial print
        for (size_t i{}; i < rows; ++i) {
          for (size_t j{}; j < cols; ++j) {
            os << mat(i, j) << " ";
          }
          os << "\n";
        }
      }
      return os;
    } else {
      static_assert(true, "[ERROR] Not a Type type");
    }
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
template <typename T, typename... Args>
Matrix<T> &CreateMatrix(Args &&...args) {
  auto tmp = Allocate<Matrix<T>>(std::forward<Args>(args)...);
  return *tmp;
}

template <typename T, typename... Args>
Matrix<T> *CreateMatrixPtr(Args &&...args) {
  auto tmp = Allocate<Matrix<T>>(std::forward<Args>(args)...);
  return tmp.get();
}
