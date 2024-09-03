/**
 * @file include/Matrix/MatrixSplOps/MatrixZeroOps.hpp
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

#include "CommonHeader.hpp"

// Is the matrix zero?
bool IsZeroMatrix(Matrix<Type>*);

// Zero matrix addition
Matrix<Type>* ZeroMatAdd(Matrix<Type>*, Matrix<Type>*);
// Zero matrix multiplication
Matrix<Type>* ZeroMatMul(Matrix<Type>*, Matrix<Type>*);
// Zero matrix kronocker product
Matrix<Type>* ZeroMatKron(Matrix<Type>*, Matrix<Type>*);

// Zero matrix addition numerics
Matrix<Type>* ZeroMatAddNum(Matrix<Type>*, Matrix<Type>*);
// Zero matrix multiplication numerics
Matrix<Type>* ZeroMatMulNum(Matrix<Type>*, Matrix<Type>*);
// Zero matrix kronocker product numerics
Matrix<Type>* ZeroMatKronNum(Matrix<Type>*, Matrix<Type>*);
