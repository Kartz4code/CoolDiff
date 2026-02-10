/**
 * @file include/Matrix/MatrixInterface/MatrixUtils.ipp
 *
 * @copyright 2023-2026 Karthik Murali Madhavan Rathai
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

// Get block matrix
template<typename T>
void Matrix<T>::getBlockMat(const Pair<size_t, size_t>& rows, const Pair<size_t, size_t>& cols, Matrix*& result) const {
    const size_t row_start = rows.first;
    const size_t row_end = rows.second;
    const size_t col_start = cols.first;
    const size_t col_end = cols.second;

    // Assert for row start/end, column start/end and index out of bound checks
    ASSERT((row_start >= 0 && row_start < m_rows), "Row starting index out of bound");
    ASSERT((row_end >= 0 && row_end < m_rows), "Row ending index out of bound");
    ASSERT((col_start >= 0 && col_start < m_cols), "Column starting index out of bound");
    ASSERT((col_end >= 0 && col_end < m_cols), "Column ending index out of bound");
    ASSERT((row_start <= row_end), "Row start greater than row ending");
    ASSERT((col_start <= col_end), "Column start greater than row ending");

    MemoryManager::MatrixPool(result, row_end - row_start + 1, col_end - col_start + 1, allocatorType());
    const auto outer_idx = CoolDiff::Common::Range<size_t>(row_start, row_end + 1);
    const auto inner_idx = CoolDiff::Common::Range<size_t>(col_start, col_end + 1);
    std::for_each(EXECUTION_PAR outer_idx.begin(), outer_idx.end(),
    [this, &col_start, &row_start, &inner_idx, result](const size_t i) {
        std::for_each(EXECUTION_PAR inner_idx.begin(), inner_idx.end(),
                    [i, this, &col_start, &row_start, &inner_idx, result](const size_t j) {
                        (*result)(i - row_start, j - col_start) = (*this)(i, j);
        });
    });
}

// Set block matrix
template<typename T>
void Matrix<T>::setBlockMat(const Pair<size_t, size_t>& rows, const Pair<size_t, size_t>& cols, const Matrix* result) {
    const size_t row_start = rows.first;
    const size_t row_end = rows.second;
    const size_t col_start = cols.first;
    const size_t col_end = cols.second;

    // Assert for row start/end, column start/end and index out of bound checks
    ASSERT((row_start >= 0 && row_start < m_rows), "Row starting index out of bound");
    ASSERT((row_end >= 0 && row_end < m_rows), "Row ending index out of bound");
    ASSERT((col_start >= 0 && col_start < m_cols), "Column starting index out of bound");
    ASSERT((col_end >= 0 && col_end < m_cols), "Column ending index out of bound");
    ASSERT((row_start <= row_end), "Row start greater than row ending");
    ASSERT((col_start <= col_end), "Column start greater than row ending");
    ASSERT((row_end - row_start + 1 == result->getNumRows()), "Row mismatch for insertion matrix");
    ASSERT((col_end - col_start + 1 == result->getNumColumns()), "Column mismatch for insertion matrix");

    // Special matrix embedding
    const auto outer_idx = CoolDiff::Common::Range<size_t>(row_start, row_end + 1);
    const auto inner_idx = CoolDiff::Common::Range<size_t>(col_start, col_end + 1);

    std::for_each(EXECUTION_PAR outer_idx.begin(), outer_idx.end(),
    [this, &col_start, &row_start, &inner_idx, result](const size_t i) {
        std::for_each(EXECUTION_PAR inner_idx.begin(), inner_idx.end(),
            [i, this, &col_start, &row_start, &inner_idx, result](const size_t j) {
            (*this)(i, j) = (*result)(i - row_start, j - col_start);
        });
    });
}

// Add zero padding
template<typename T>
void Matrix<T>::pad(const size_t r, const size_t c, Matrix*& result) const {
    // Special matrix embedding
    MemoryManager::MatrixPool(result, m_rows + 2 * r, m_cols + 2 * c, allocatorType());
    result->setBlockMat({r, r + m_rows - 1}, {c, c + m_cols - 1}, this);
}

// Set values for the result matrix
template<typename T>
void Matrix<T>::setEval() {
    if ((nullptr != getMatrixPtr()) && (nullptr != mp_result) && (nullptr != mp_result->getMatrixPtr())) {
        std::transform(EXECUTION_SEQ getMatrixPtr(), getMatrixPtr() + getNumElem(), mp_result->getMatrixPtr(), 
                        [this](auto& v) { return CoolDiff::TensorR1::Eval(v); });
    }
}

// Set value for the derivative result matrix
template<typename T>
void Matrix<T>::setDevalF(const Matrix<Variable>& X) {
    // If the matrix type is Expression
    if constexpr (true == std::is_same_v<T, Expression>) {
        if ((nullptr != getMatrixPtr()) && (nullptr != mp_dresult) && (nullptr != mp_dresult->getMatrixPtr())) {
            // Precompute the reverse derivatives
            std::for_each(EXECUTION_SEQ getMatrixPtr(), getMatrixPtr() + getNumElem(), [](auto& i) { CoolDiff::TensorR1::PreComp(i); });

            // Get dimensions of X variable matrix
            const size_t xrows = X.getNumRows();
            const size_t xcols = X.getNumColumns();

            // Vector of indices in current matrix
            const auto outer_idx = CoolDiff::Common::Range<size_t>(0, getNumElem());
            const auto inner_idx = CoolDiff::Common::Range<size_t>(0, (xrows * xcols));

            // Logic for Kronecker product (Reverse mode differentiation)
            std::for_each(EXECUTION_PAR outer_idx.begin(), outer_idx.end(),
                    [this, &X, &inner_idx, xrows, xcols](const size_t i) {
                    const size_t k = (i % m_cols);
                    const size_t l = ((i - k) / m_cols);

                    // Inner loop
                    std::for_each(EXECUTION_PAR inner_idx.begin(), inner_idx.end(), [&](const size_t n) {
                            const size_t j = (n % xcols);
                            const size_t i = ((n - j) / xcols);
                            (*mp_dresult)((l * xrows) + i, (k * xcols) + j) = CoolDiff::TensorR1::DevalR((*this)(l, k), X(i, j));
                    });
            });
        }
    }
    // If the matrix type is Variable
    else if constexpr (true == std::is_same_v<T, Variable>) {
        // Last is the most important condition to delineate Matrix<Variable>'s
        if ((nullptr != getMatrixPtr()) && (nullptr != mp_dresult) && (nullptr != mp_dresult->getMatrixPtr()) && (m_nidx == X.m_nidx)) {
            // Get dimensions of X variable matrix
            const size_t xrows = X.getNumRows();
            const size_t xcols = X.getNumColumns();

            // Vector of indices in X matrix
            const auto idx = CoolDiff::Common::Range<size_t>(0, X.getNumElem());
            // Logic for Kronecker product (With ones)
            std::for_each(EXECUTION_PAR idx.begin(), idx.end(),
                    [this, xrows, xcols](const size_t n) {
                        const size_t j = (n % xcols);
                        const size_t i = ((n - j) / xcols);
                        // Inner loop
                        (*mp_dresult)((i * xrows) + i, (j * xcols) + j) = (Type)(1);
            });
        }
        // If the Matrix is Variable, but of different form
        else if ((nullptr != getMatrixPtr()) &&  (nullptr != mp_dresult) && (nullptr != mp_dresult->getMatrixPtr())) {
            // Get dimensions of X variable matrix
            const size_t xrows = X.getNumRows();
            const size_t xcols = X.getNumColumns();

            // Vector of indices in current matrix
            const auto outer_idx = CoolDiff::Common::Range<size_t>(0, getNumElem());
            const auto inner_idx = CoolDiff::Common::Range<size_t>(0, xrows * xcols);

            // Logic for Kronecker product (Forward mode differentiation)
            std::for_each(EXECUTION_PAR outer_idx.begin(), outer_idx.end(),
                    [this, &X, &inner_idx, xrows, xcols](const size_t i) {
                    const size_t k = (i % m_cols);
                    const size_t l = ((i - k) / m_cols);

                    // Inner loop
                    std::for_each(EXECUTION_PAR inner_idx.begin(), inner_idx.end(),
                            [&](const size_t n) {
                                const size_t j = (n % xcols);
                                const size_t i = (n - j) / xcols;
                                (*mp_dresult)((l * xrows) + i, (k * xcols) + j) = (((*this)(l, k).m_nidx == X(i, j).m_nidx) ? (Type)(1) : (Type)(0));
                            });
            });
        }
    }
}

// Find me
template<typename T>
bool Matrix<T>::findMe(void* v) const {
    if (static_cast<const void*>(this) == v) {
        return true;
    } else {
        return false;
    }
}

// Reset impl
template<typename T>
void Matrix<T>::resetImpl() {
    // Reset flag
    this->m_visited = true;

    // Reset states
    m_eval = false;
    m_devalf = false;

    // Empty matrix cache
    if (false == m_cache.empty()) {
        m_cache.clear();
    }

    // For each element
    std::for_each(EXECUTION_SEQ m_gh_vec.begin(), m_gh_vec.end(),
                    [](auto*& item) {
                        if (nullptr != item) {
                            item->reset();
                        }
                    });

    this->m_visited = false;
}

// Matrix reshape
template<typename T>
void Matrix<T>::reshape(const size_t rows, const size_t cols) {
    ASSERT((rows*cols == getNumElem()), "Number of elements not the same between reshape and actual matrix");
    m_rows = rows; m_cols = cols;
}

// To output stream
template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat) { 
    const size_t rows = mat.getFinalNumRows();
    const size_t cols = mat.getFinalNumColumns();
    std::ostringstream oss;
    if constexpr (true == std::is_same_v<T, Type>) {
        // Serial print
        for (size_t i{}; i < rows; ++i) {
            for (size_t j{}; j < cols; ++j) {
                oss << mat(i, j) << " ";
            }
            oss << "\n";
        }
        return { os << oss.str() };
    } else if constexpr (true == std::is_arithmetic_v<T>) {
        // Serial print
        for (size_t i{}; i < rows; ++i) {
            for (size_t j{}; j < cols; ++j) {
                oss << mat(i, j) << " ";
            }
            oss << "\n";
        }
        return { os << oss.str() };
    } else {
        ASSERT(false, "Matrix not in printable format");
    }
}