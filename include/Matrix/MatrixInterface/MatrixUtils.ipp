 #pragma once

/**
 * @file include/Matrix/MatrixInterface/MatrixUtils.ipp
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

// Set values for the result matrix
template<typename T>
void Matrix<T>::setEval() {
    if ((nullptr != mp_mat) && (nullptr != mp_result) && (nullptr != mp_result->mp_mat)) {
        std::transform(EXECUTION_SEQ mp_mat, mp_mat + getNumElem(), mp_result->mp_mat, 
                        [this](auto& v) { return CoolDiff::Scalar::Eval(v); });
    }
}

// Set value for the derivative result matrix
template<typename T>
void Matrix<T>::setDevalF(const Matrix<Variable>& X) {
    // If the matrix type is Expression
    if constexpr (true == std::is_same_v<T, Expression>) {
        if ((nullptr != mp_mat) && (nullptr != mp_dresult) && (nullptr != mp_dresult->mp_mat)) {
            // Precompute the reverse derivatives
            std::for_each(EXECUTION_SEQ mp_mat, mp_mat + getNumElem(), 
                          [](auto& i) { CoolDiff::Scalar::PreComp(i); });

            // Get dimensions of X variable matrix
            const size_t xrows = X.getNumRows();
            const size_t xcols = X.getNumColumns();

            // Vector of indices in current matrix
            const auto outer_idx = Range<size_t>(0, getNumElem());
            const auto inner_idx = Range<size_t>(0, (xrows * xcols));

            // Logic for Kronecker product (Reverse mode differentiation)
            std::for_each(EXECUTION_PAR outer_idx.begin(), outer_idx.end(),
                    [this, &X, &inner_idx, xrows, xcols](const size_t i) {
                    const size_t k = (i % m_cols);
                    const size_t l = ((i - k) / m_cols);

                    // Inner loop
                    std::for_each(EXECUTION_PAR inner_idx.begin(), inner_idx.end(), [&](const size_t n) {
                            const size_t j = (n % xcols);
                            const size_t i = ((n - j) / xcols);
                            (*mp_dresult)((l * xrows) + i, (k * xcols) + j) = CoolDiff::Scalar::DevalR((*this)(l, k), X(i, j));
                    });
            });
        }
    }
    // If the matrix type is Variable
    else if constexpr (true == std::is_same_v<T, Variable>) {
        // Last is the most important condition to delineate Matrix<Variable>'s
        if ((nullptr != mp_mat) && (nullptr != mp_dresult) && (nullptr != mp_dresult->mp_mat) && (m_nidx == X.m_nidx)) {
            // Get dimensions of X variable matrix
            const size_t xrows = X.getNumRows();
            const size_t xcols = X.getNumColumns();

            // Vector of indices in X matrix
            const auto idx = Range<size_t>(0, X.getNumElem());
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
        else if ((nullptr != mp_mat) &&  (nullptr != mp_dresult) && (nullptr != mp_dresult->mp_mat)) {
            // Get dimensions of X variable matrix
            const size_t xrows = X.getNumRows();
            const size_t xcols = X.getNumColumns();

            // Vector of indices in current matrix
            const auto outer_idx = Range<size_t>(0, getNumElem());
            const auto inner_idx = Range<size_t>(0, xrows * xcols);

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

// To output stream
template<typename T>
std::ostream &operator<<(std::ostream& os, const Matrix<T>& mat) {
    const size_t rows = mat.getFinalNumRows();
    const size_t cols = mat.getFinalNumColumns();
    if constexpr (true == std::is_same_v<T, Type>) {
        if (mat.getMatType() == MatrixSpl::ZEROS) {
            os << "Zero matrix of dimension: " << "(" << rows << "," << cols << ")\n";
        } else if (mat.getMatType() == MatrixSpl::EYE) {
            os << "Identity matrix of dimension: " << "(" << rows << "," << cols << ")\n";
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
    } else if constexpr (true == std::is_arithmetic_v<T>) {
        // Serial print
        for (size_t i{}; i < rows; ++i) {
            for (size_t j{}; j < cols; ++j) {
                os << mat(i, j) << " ";
            }
            os << "\n";
        }
        return os;
    } else {
        ASSERT(false, "Matrix not in printable format");
    }
}