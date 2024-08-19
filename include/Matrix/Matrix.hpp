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

#include "IMatrix.hpp"

// Factory function for matrix reference creation
template <typename T>
Matrix<T> &CreateMatrix(size_t, size_t);

// Factory function for matrix pointer creation
template <typename T>
Matrix<T> *CreateMatrixPtr(size_t, size_t);

// Matrix class
template <typename T>
class Matrix : public IMatrix<Matrix<T>>
{
private:
    // Matrix row and column size
    size_t m_rows{0};
    size_t m_cols{0};

    // Collection of meta variable expressions
    Vector<MetaMatrix *> m_gh_vec{};

    inline constexpr Type getValue(T &val) const
    {
        // If T is of type Type
        if constexpr (true == std::is_same_v<T, Type>)
        {
            return val;
            // If T is of type Variable
        }
        else if constexpr (true == std::is_same_v<T, Variable> ||
                           true == std::is_same_v<T, Parameter>)
        {
            return val.getValue();
            // If T is of type Expression
        }
        else if constexpr (true == std::is_same_v<T, Expression>)
        {
            return Eval(val);
        }
        else
        {
            // If T is unknown, then return 0
            return (Type)(0);
        }
    }

public:
    // Matrix raw pointer of underlying type (Expression, Variable, Parameter, Type)
    T *m_mat{nullptr};
    // Matrix pointer for evaluation result (Type)
    Matrix<Type> *m_result{nullptr};

    // Block index
    size_t m_nidx{};

    // Default constructor
    Matrix()
        : m_rows{0}, m_cols{0}, m_mat{new T{}}, m_result{nullptr},
          m_nidx{this->m_idx_count++}
    {
    }

    // Initalize the matrix with rows and columns
    Matrix(size_t rows, size_t cols)
        : m_rows{rows}, m_cols{cols}, m_mat{new T[rows * cols]{}},
          m_result{nullptr}, m_nidx{this->m_idx_count++}
    {
    }

    // Matrix expressions constructor
    template <typename Z>
    Matrix(const IMatrix<Z> &expr)
        : m_rows{expr.getNumRows()}, m_cols{expr.getNumColumns()},
          m_mat{new T[m_rows * m_cols]{}}, m_result{nullptr},
          m_nidx{this->m_idx_count++}
    {
        // Reserve a buffer of Matrix expressions
        m_gh_vec.reserve(g_vec_init);
        // Emplace the expression in a generic holder
        m_gh_vec.emplace_back(&static_cast<const Z &>(expr));
    }

    /* Copy assignment for expression evaluation */
    template <typename Z>
    Matrix &operator=(const IMatrix<Z> &expr)
    {
        if (static_cast<const Z &>(expr).findMe(this) == false)
        {
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
          m_mat{std::exchange(m.m_mat, nullptr)}, m_result{std::exchange(
                                                      m.m_result,
                                                      nullptr)},
          m_gh_vec{std::exchange(m.m_gh_vec, {})}, m_nidx{
                                                       std::exchange(m.m_nidx,
                                                                     -1)}
    {
    }

    // Move assignment operator
    Matrix &operator=(Matrix &&m) noexcept
    {
        if (nullptr != m_mat)
        {
            delete[] m_mat;
        }

        // Exchange values
        m_mat = std::exchange(m.m_mat, nullptr);
        m_rows = std::exchange(m.m_rows, -1);
        m_cols = std::exchange(m.m_cols, -1);
        m_result = std::exchange(m.m_result, nullptr);
        m_nidx = std::exchange(m.m_nidx, -1);
        m_gh_vec = std::exchange(m.m_gh_vec, {});

        // Return this reference
        return *this;
    }

    // Copy constructor
    Matrix(const Matrix &m)
        : m_rows{m.m_rows}, m_cols{m.m_cols}, m_mat{new T[m_rows * m_cols]{}},
          m_nidx{m.m_nidx}, m_gh_vec{m.m_gh_vec}
    {
        // Copy values
        std::copy(m.m_mat, m.m_mat + m_rows * m_cols, m_mat);
        if (nullptr != m.m_result)
        {
            m_result = m.m_result->clone();
        }
    }

    // Copy assignment operator
    Matrix &operator=(const Matrix &m)
    {
        if (&m != this)
        {
            // Assign resources
            m_rows = m.m_rows;
            m_cols = m.m_cols;
            m_nidx = m.m_nidx;
            m_gh_vec = m.m_gh_vec;

            // Copy m_mat
            if (nullptr != m_mat)
            {
                delete[] m_mat;
            }
            m_mat = new T[m_rows * m_cols]{};
            std::copy(m.m_mat, m.m_mat + m_rows * m_cols, m_mat);

            // Copy m_result
            if (nullptr != m_result)
            {
                delete[] m_result;
            }
            m_result = m.m_result->clone();
        }

        // Return this reference
        return *this;
    }

    // For cloning numerical values for copy constructors
    inline Matrix<Type> *clone() const
    {
        Matrix<Type> *result = CreateMatrixPtr<Type>(m_rows, m_cols);
        if (nullptr != result->m_mat)
        {
            std::copy(m_mat, m_mat + m_rows * m_cols, result->m_mat);
        }
        return result;
    }

    // Reshape matrix (rows and columns)
    void reshape(size_t rows, size_t cols)
    {
        assert((rows * cols == m_rows * m_cols) &&
               "[ERROR] Matrix resize operation invalid");
        // Assign new rows and cols
        m_rows = rows;
        m_cols = cols;
    }

    // Get matrix pointer const
    const T *getMatrixPtr() const
    {
        return m_mat;
    }

    // Get matrix pointer
    T *getMatrixPtr()
    {
        return m_mat;
    }

    // Matrix 2D access using operator()()
    T &operator()(const size_t i, const size_t j)
    {
        return m_mat[i * m_cols + j];
    }

    // Matrix 2D access using operator()() const
    const T &operator()(const size_t i, const size_t j) const
    {
        return m_mat[i * m_cols + j];
    }

    // Matrix 1D access using operator[]
    T &operator[](const size_t l)
    {
        return m_mat[l];
    }

    // Matrix 1D access using operator[] const
    const T &operator[](const size_t l) const
    {
        return m_mat[l];
    }

    // Get a row (Move)
    Matrix getRow(const size_t &i) &&
    {
        Matrix tmp(m_cols, 1);
        std::copy(m_mat + i * m_cols,
                  m_mat + (i + 1) * m_cols,
                  tmp.getMatrixPtr());
        return std::move(tmp);
    }

    // Get a row (Copy)
    Matrix getRow(const size_t &i) const &
    {
        Matrix tmp(m_cols, 1);
        std::copy(m_mat + i * m_cols,
                  m_mat + (i + 1) * m_cols,
                  tmp.getMatrixPtr());
        return tmp;
    }

    // Get a column (Move)
    Matrix getColumn(const size_t &i) &&
    {
        Matrix tmp(m_rows, 1);
        for (size_t j{}; j < m_rows; ++j)
        {
            tmp.m_mat[j] = m_mat[j * m_rows + i];
        }
        return std::move(tmp);
    }

    // Get a column (copy)
    Matrix getColumn(const size_t &i) const &
    {
        Matrix tmp(m_rows, 1);
        for (size_t j{}; j < m_rows; ++j)
        {
            tmp.m_mat[j] = m_mat[j * m_rows + i];
        }
        return tmp;
    }

    // Get number of rows
    V_OVERRIDE(size_t getNumRows() const)
    {
        return m_rows;
    }

    // Get number of columns
    V_OVERRIDE(size_t getNumColumns() const)
    {
        return m_cols;
    }

    V_OVERRIDE(Matrix<Type> *eval())
    {
        // Cache the m_result value
        if (false == this->m_visited)
        {
            if (m_result == nullptr)
            {
                m_result = CreateMatrixPtr<Type>(m_rows, m_cols);
                for (size_t i{0}; i < m_rows * m_cols; ++i)
                {
                    (*m_result)[i] = getValue(m_mat[i]);
                }
            }

            // Set visit flag to true
            this->m_visited = true;
            // Loop on internal equations
            for (auto &i : m_gh_vec)
            {
                if (nullptr != i)
                {
                    m_result = i->eval();
                }
            }
        }
        return m_result;
    }

    // Reset all visited flags
    V_OVERRIDE(void reset())
    {
        if (true == this->m_visited)
        {
            this->m_visited = false;
            for (auto &i : m_gh_vec)
            {
                if (nullptr != i)
                {
                    i->reset();
                }
            }
        }
        // Reset flag
        this->m_visited = false;
    }

    // Find me
    V_OVERRIDE(bool findMe(void *v) const)
    {
        if (static_cast<const void *>(this) == v)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    // Get type
    V_OVERRIDE(std::string_view getType() const)
    {
        return "Matrix";
    }

    // Reset impl
    inline void resetImpl()
    {
        this->m_visited = true;
        // Reset states
        for (auto &i : m_gh_vec)
        {
            if (i != nullptr)
            {
                i->reset();
            }
        }
        this->m_visited = false;
    }

    // To output stream
    friend std::ostream &operator<<(std::ostream &os, const Matrix &mat)
    {
        for (size_t i{0}; i < mat.m_rows * mat.m_cols; ++i)
        {
            os << mat.getValue(mat.m_mat[i]) << " ";
        }
        os << "\n";
        return os;
    }

    V_DTR(~Matrix())
    {
        // If m_mat is not nullptr, delete it
        if (nullptr != m_mat)
        {
            delete[] m_mat;
            m_mat = nullptr;
        }
    }
};

// Factory function for matrix creation
template <typename T>
Matrix<T> &CreateMatrix(size_t rows, size_t cols)
{
    auto tmp = Allocate<Matrix<T>>(rows, cols);
    return *tmp;
}

template <typename T>
Matrix<T> *CreateMatrixPtr(size_t rows, size_t cols)
{
    auto tmp = Allocate<Matrix<T>>(rows, cols);
    return tmp.get();
}