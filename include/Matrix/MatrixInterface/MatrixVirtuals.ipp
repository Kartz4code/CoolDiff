 #pragma once

/**
 * @file include/Matrix/MatrixInterface/MatrixVirtuals.ipp
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

// Evaluate matrix
template<typename T>
Matrix<Type>* Matrix<T>::eval() {
    // Cache the mp_result value
    if constexpr (false == std::is_same_v<T, Type>) {
        if (nullptr != mp_mat) {
            MemoryManager::MatrixPool(m_rows, m_cols, mp_result);
        }
    } else {
        MemoryManager::MatrixPool(m_rows, m_cols, mp_result);
    }

    // If value not evaluated, compute it again
    if (false == m_eval) {
        setEval(); m_eval = true;
    }

    // If visited already
    if (false == this->m_visited) {
        // Set visit flag to true
        this->m_visited = true;
        // Loop on internal equations
        std::for_each(EXECUTION_SEQ m_gh_vec.begin(), m_gh_vec.end(),
                    [this](auto* i) {
                        if (nullptr != i) {
                            mp_result = i->eval(); m_eval = true;
                            m_rows = i->getNumRows();
                            m_cols = i->getNumColumns();
                        }
                    });
    }

    // Return evaulation result
    return mp_result;
}

// Derivative matrix
template<typename T>
Matrix<Type>* Matrix<T>::devalF(Matrix<Variable>& X) {
    // Derivative result computation
    const size_t xrows = X.getNumRows();
    const size_t xcols = X.getNumColumns();
    if constexpr (true == std::is_same_v<T, Type> || true == std::is_same_v<T, Parameter>) {
        #if defined(NAIVE_IMPL)
            mp_dresult = MemoryManager::MatrixSplPool((m_rows * xrows), (m_cols * xcols), MatrixSpl::ZEROS);
        #else
            MemoryManager::MatrixPool((m_rows * xrows), (m_cols * xcols), mp_dresult);
        #endif
    } else {
        if (nullptr != mp_mat) {
            MemoryManager::MatrixPool((m_rows * xrows), (m_cols * xcols), mp_dresult);
        }
    }

    // If derivative not evaluated, compute it again
    if (false == m_devalf) {
        setDevalF(X); m_devalf = true;
    }

    // If visited already
    if (false == this->m_visited) {
        // Set visit flag to true
        this->m_visited = true;
        
        // Loop on internal equations
        std::for_each(EXECUTION_SEQ m_gh_vec.begin(), m_gh_vec.end(),
                    [this, &X](auto* i) {
                        if (nullptr != i) {
                            mp_dresult = i->devalF(X); m_devalf = true;
                            mp_result = i->eval(); m_eval = true;
                            m_rows = i->getNumRows();
                            m_cols = i->getNumColumns();
                        }
                    });
    }

    // Return derivative result
    return mp_dresult;
}

// Traverse tree
template<typename T>
void Matrix<T>::traverse(OMMatPair* cache) {
  if (false == this->m_visited) {
    this->m_visited = true;
    // Reset states
    std::for_each(EXECUTION_SEQ m_gh_vec.begin(), m_gh_vec.end(),
                    [this, &cache](auto* i) {
                        if (nullptr != i) {
                            // Traverse the tree
                            i->traverse();
                            // Set value
                            mp_result = i->eval();  m_eval = true;
                            // Save cache
                            m_cache = i->getCache();
                            m_rows = i->getNumRows();
                            m_cols = i->getNumColumns();
                        }
                    });
  }
}

// Reset all visited flags
template<typename T>
void Matrix<T>::reset() {
    if (true == this->m_visited) {
        this->m_visited = false;
        // Reset states
        m_eval = false;
        m_devalf = false;

        // For each element
        std::for_each(EXECUTION_SEQ m_gh_vec.begin(), m_gh_vec.end(), 
                    [](auto* item) {
                        if (nullptr != item) {
                            item->reset();
                        }
                    });
    }
    // Reset flag
    this->m_visited = false;

    // Free all results
    if (mp_result != nullptr) {
        mp_result->free();
    }

    // Free all derivative results
    if (mp_dresult != nullptr) {
        mp_dresult->free();
    }

    // Empty cache
    if (false == m_cache.empty()) {
        m_cache.clear();
    }
}