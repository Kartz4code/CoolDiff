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

// Matrix class
template<typename T>
class Matrix : public IMatrix<Matrix<T>> {
private:
    // Matrix resources
    T* m_mat{nullptr};
    size_t m_rows{ 0 }, m_cols{ 0 };

public:
    // Default constructor
    Matrix() : m_rows{ 0 }, m_cols{ 0 } {
        m_mat = new T{T(0)};
    }

    // Initalize the matrix
    Matrix(size_t rows, size_t cols) : m_rows{ rows }, m_cols{ cols } {
        m_mat = new T[rows*cols]{T(0)};
    }

    // Move constructor 
    Matrix(Matrix&& m) noexcept : m_mat{std::exchange(m.m_mat, nullptr)},
                                  m_rows{std::exchange(m.m_rows,-1)},
                                  m_cols{std::exchange(m.m_cols,-1)}
    {}
    
    // Move assignment 
    Matrix& operator=(Matrix&& m) noexcept {
        m_mat = std::exchange(m.m_mat, nullptr);
        m_rows = std::exchange(m.m_rows,-1);
        m_cols = std::exchange(m.m_cols,-1); 
        return *this;    
    }

    // Copy assignment 
    Matrix& operator=(const Matrix& m) {
        if(&m != this) {
            m_rows = m.m_rows;
            m_cols = m.m_cols;
            if(nullptr != m_mat) {
                delete[] m_mat; 
            }
            m_mat = new T[m_rows*m_cols]{T(0)};
            std::copy(m.m_mat, m.m_mat+m_rows*m_cols, m_mat);
        }
        return *this;
    }

    // Copy constructor 
    Matrix(const Matrix& m) : m_rows{m.m_rows},
                              m_cols{m.m_cols} {
        m_mat = new T[m_rows*m_cols]{T(0)};
        std::copy(m.m_mat, m.m_mat+m_rows*m_cols, m_mat);
    }

    // Resizing matrix (rows and columns) 
    void resize(size_t rows, size_t cols) {
        assert((rows*cols == m_rows*m_cols) && "[ERROR] Matrix resize operation invalid");
        m_rows = rows; m_cols = cols;
    }

    // Get matrix pointer const
    const T* getMatrixPtr() const {
        return m_mat;
    }

    // Get matrix pointer
    T* getMatrixPtr() {
        return m_mat;
    }

    // Get matrix element
    T getMatrixElem(size_t i, size_t j) const { 
        return m_mat[i*m_cols + j]; 
    }
    
    // Set matrix element
    void setMatrixElem(size_t i, size_t j, const T& val) { 
        m_mat[i*m_cols + j] = val; 
    }

    // Matrix access using operator()()
    T& operator()(size_t i, size_t j) {
        return m_mat[i*m_cols + j]; 
    }

    // Matrix access using operator()() const
    const T& operator()(size_t i, size_t j) const {
        return m_mat[i*m_cols + j]; 
    }

    // Get number of rows
    size_t getNumRows() const { 
        return m_rows; 
    }

    // Get number of columns
    size_t getNumColumns() const { 
        return m_cols; 
    }

    // Get a row (Move)
    Matrix getRow(const size_t& i) && {
        Matrix tmp(m_cols, 1);
        std::copy(m_mat+i*m_cols, m_mat+(i+1)*m_cols, tmp.getMatrixPtr());
        return std::move(tmp);
    }

    // Get a row (Copy)
    Matrix getRow(const size_t& i) const& {
        Matrix tmp(m_cols, 1);
        std::copy(m_mat+i*m_cols, m_mat+(i+1)*m_cols, tmp.getMatrixPtr());
        return tmp;
    }

    // Get a column (Move)
    Matrix getColumn(const size_t& i) && {
        Matrix tmp(m_rows, 1);
        for(size_t j{}; j < m_rows; ++j) {
            tmp.m_mat[j] = m_mat[j*m_rows + i]; 
        }
        return std::move(tmp);
    }

    // Get a column (copy)
    Matrix getColumn(const size_t& i) const & {
        Matrix tmp(m_rows, 1);
        for(size_t j{}; j < m_rows; ++j) {
            tmp.m_mat[j] = m_mat[j*m_rows + i]; 
        }
        return tmp;
    }

    V_OVERRIDE(void eval(MetaMatrix&)) {
        return;
    }

    // Reset all visited flags
    V_OVERRIDE(void reset()) {
        return;
    }

    // Find me
    V_OVERRIDE(bool findMe(void*) const) {
        return true;
    }

    // Get type
    V_OVERRIDE(std::string_view getType() const) {
        return "Matrix";
    }

    // To output stream
    friend std::ostream& operator<<(std::ostream& os, const  Matrix<T>& mat) {
        for (size_t i{ 0 }; i < mat.m_rows*mat.m_cols; ++i) {
            os << mat.m_mat[i] << " ";
        }
        os << "\n";
        return os;
    }

    V_DTR(~Matrix()) {
        delete[] m_mat;
    }
};