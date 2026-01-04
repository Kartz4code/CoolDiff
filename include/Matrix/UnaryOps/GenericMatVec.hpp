/**
 * @file include/Matrix/UnaryOps/GenericMatVec.hpp
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

// Left/right side is a Matrix
template <typename T>
class GenericMatVec : public IMatrix<GenericMatVec<T>> {
    private:
        // Resources
        T* mp_right{nullptr};

        // Disable copy and move constructors/assignments
        #if 0
            DISABLE_COPY(GenericMatVec)
            DISABLE_MOVE(GenericMatVec)
        #endif

        // All matrices
        inline static constexpr const size_t m_size{2};
        Matrix<Type>* mp_arr[m_size]{};

    public:
        // Block index
        const size_t m_nidx{};
        // Cache for reverse AD 1st
        OMMatPair m_cache;

        // Constructor
        constexpr GenericMatVec(T* u) : mp_right{u}, m_nidx{this->m_idx_count++} {
            std::fill_n(EXECUTION_PAR mp_arr, m_size, nullptr);
        }

        // Get number of rows
        V_OVERRIDE(size_t getNumRows() const) { 
            const size_t nrows = mp_right->getNumRows();
            const size_t ncols = mp_right->getNumColumns();       
            return (nrows*ncols); 
        }

        // Get number of columns
        V_OVERRIDE(size_t getNumColumns() const) { 
            return 1; 
        }

        // Find me
        bool findMe(void* v) const { 
            BINARY_RIGHT_FIND_ME(); 
        }

    // Clone matrix expression
        constexpr const auto& cloneExp() const {
            return vec(*mp_right);
        }

        // Matrix eval computation
        V_OVERRIDE(Matrix<Type>* eval()) {
            // Get number of rows/columns 
            const size_t nrows = getNumRows();
            const size_t ncols = getNumColumns();

            // Reshape and return
            Matrix<Type>* right_mat = mp_right->eval();
            right_mat->reshape(nrows, ncols);
            return right_mat;
        }

        // Matrix devalF computation
        V_OVERRIDE(Matrix<Type>* devalF(Matrix<Variable>& X)) {
            // Get number of rows/columns 
            const size_t xrows = X.getNumRows(); 
            const size_t xcols = X.getNumColumns();
            const size_t nrows = getNumRows();
            const size_t ncols = getNumColumns();

            // Matrix derivative vectorization
            Matrix<Variable> Y(1, xrows*xcols, X.getMatrixPtr());

            // Reshape and return
            Matrix<Type>* dright_mat = mp_right->devalF(Y);
            dright_mat->reshape(nrows*xrows, ncols*xcols);
            return dright_mat;
        }

        // Traverse
        V_OVERRIDE(void traverse(OMMatPair* cache = nullptr)) {
            // TODO - Vec traverse
        }

        // Get cache
        V_OVERRIDE(OMMatPair& getCache()) {
            return m_cache;
        }

        // Reset visit run-time
        V_OVERRIDE(void reset()) { 
            BINARY_MAT_RIGHT_RESET(); 
        }

        // Get type
        V_OVERRIDE(std::string_view getType() const) { 
            return "GenericMatVec"; 
        }

        // Destructor
        V_DTR(~GenericMatVec()) = default;
};

// GenericMatVec with 1 typename and callables
template <typename T> 
using GenericMatVecT = GenericMatVec<T>;

// Function for vectorize computation
template <typename T> 
constexpr const auto& vec(const IMatrix<T>& u) {
    const auto& _u = u.cloneExp();
    auto tmp = Allocate<GenericMatVecT<T>>(const_cast<T*>(static_cast<const T*>(&_u)));
    return *tmp;
}