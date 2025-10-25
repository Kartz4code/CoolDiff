/**
 * @file include/Matrix/MatrixHandler/MatrixSubHandler/NaiveCPU/EyeMatSubHandler.hpp
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

 #include "MatrixStaticHandler.hpp"
 
 #include "Matrix.hpp"
 #include "MatrixZeroOps.hpp"
 
 template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
 class EyeMatSubHandler : public T {
     public:
        void handle(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
            #if defined(USE_SYMBOLIC_CHECK)
                // Dimensions of LHS and RHS matrices
                const size_t nrows{lhs->getNumRows()};
                const size_t ncols{rhs->getNumColumns()};
                const size_t lcols{lhs->getNumColumns()};
                const size_t rrows{rhs->getNumRows()};

                // Assert dimensions
                ASSERT((nrows == rrows) && (ncols == lcols), "Matrix subtraction dimensions mismatch");

                /* Zero matrix special check */
                if (auto *it = EyeMatSub(lhs, rhs); nullptr != it) {
                    if (it == lhs) {
                    if (-1 == it->getMatType()) {
                        BaselineCPU::SubEyeRHS(it, result);
                    } else {
                        T::handle(lhs, rhs, result);
                    }
                    } else if (it == rhs) {
                    if (-1 == it->getMatType()) {
                        BaselineCPU::SubEyeLHS(it, result);
                    } else {
                        T::handle(lhs, rhs, result);
                    }
                    } else {
                        result = const_cast<Matrix<Type> *>(it);
                    }
                    return;
                }
                /* Zero matrix numerical check */
                #if defined(USE_NUMERICAL_CHECK)
                    else if (auto *it = EyeMatSubNum(lhs, rhs); nullptr != it) {
                        if (it == lhs) {
                        if (-1 == it->getMatType()) {
                            BaselineCPU::SubEyeRHS(it, result);
                        } else {
                            T::handle(lhs, rhs, result);
                        }
                        } else if (it == rhs) {
                        if (-1 == it->getMatType()) {
                            BaselineCPU::SubEyeLHS(it, result);
                        } else {
                            T::handle(lhs, rhs, result);
                        }
                        } else {
                            result = const_cast<Matrix<Type> *>(it);
                        }
                        return;
                    }
                #endif
            #endif

            // Chain of responsibility
            T::handle(lhs, rhs, result);
        }
 };