/**
 * @file src/Matrix/MatrixHandler/EyeMatAddHandler.cpp
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

#include "EyeMatAddHandler.hpp"
#include "MatrixEyeOps.hpp"
#include "Matrix.hpp"

void EyeMatAddHandler::handle(Matrix<Type>* lhs, 
                              Matrix<Type>* rhs, 
                              Matrix<Type>*& result) {
    // Null pointer check
    NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
    NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

    /* Eye matrix special check */ 
    if (auto* it = EyeMatAdd(lhs, rhs); nullptr != it) {     
        // Rows and columns of result matrix and if result is nullptr, then create a new resource
        const size_t nrows{lhs->getNumRows()};
        const size_t ncols{rhs->getNumColumns()};
        if (nullptr == result) {
            result = CreateMatrixPtr<Type>(nrows, ncols);
        }

        // Diagonal indices
        Vector<size_t> elem(nrows);
        std::iota(elem.begin(), elem.end(), 0);

        // Case when both left and right matrices are eye
        if(it->getMatType() == MatrixSpl::EYE) {
                std::for_each(EXECUTION_PAR 
                            elem.begin(), elem.end(), 
                            [&](const size_t i) {
                                (*result)(i,i) = (Type)(2);
                            });
        } 
        // Case when either left or right matrix is eye
        else {
                std::for_each(EXECUTION_PAR 
                            elem.begin(), elem.end(), 
                            [&](const size_t i) {
                                (*result)(i,i) = (*it)(i,i) + (Type)(1);
                            });
        }
        return;
    } 
    /* Eye matrix numerical check */
    else if (auto* it = EyeMatAddNum(lhs, rhs); nullptr != it) {
        // Rows and columns of result matrix and if result is nullptr, then create a new resource
        const size_t nrows{lhs->getNumRows()};
        const size_t ncols{rhs->getNumColumns()};
        if (nullptr == result) {
            result = CreateMatrixPtr<Type>(nrows, ncols);
        }

        // Diagonal indices
        Vector<size_t> elem(nrows);
        std::iota(elem.begin(), elem.end(), 0);
        
        // For each element
        std::for_each(EXECUTION_PAR 
                        elem.begin(), elem.end(), 
                        [&](const size_t i) {
                            (*result)(i,i) = (*lhs)(i,i) + (*rhs)(i,i);
                        });
        return;
    }

    // Chain of responsibility 
    MatrixHandler::handle(lhs, rhs, result);
}