/**
 * @file include/Matrix/MatrixSplOps/MatrixEyeOps.hpp
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

// Is the matrix identity?
bool IsEyeMatrix(Matrix<Type> *);

// Eye matrix addition
Matrix<Type> *EyeMatAdd(Matrix<Type> *, Matrix<Type> *);
// Eye matrix subtraction
Matrix<Type> *EyeMatSub(Matrix<Type> *, Matrix<Type> *);
// Eye matrix multiplication
Matrix<Type> *EyeMatMul(Matrix<Type> *, Matrix<Type> *);
// Eye matrix Kronocker product
Matrix<Type> *EyeMatKron(Matrix<Type> *, Matrix<Type> *);

// Eye matrix addition numerics
Matrix<Type> *EyeMatAddNum(Matrix<Type> *, Matrix<Type> *);
// Eye matrix subtraction numerics
Matrix<Type> *EyeMatSubNum(Matrix<Type> *, Matrix<Type> *);
// Eye matrix multiplication numerics
Matrix<Type> *EyeMatMulNum(Matrix<Type> *, Matrix<Type> *);
// Eye matrix Kronocker product numerics
Matrix<Type> *EyeMatKronNum(Matrix<Type> *, Matrix<Type> *);
