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
bool IsEyeMatrix(Matrix<Type>*);


// Eye matrix addition checks
Matrix<Type>* EyeMatAdd(Matrix<Type>*, Matrix<Type>*);
// Eye matrix multiplication checks
Matrix<Type>* EyeMatMul(Matrix<Type>*, Matrix<Type>*);

// Eye matrix addition numerical checks
Matrix<Type>* EyeMatAddNum(Matrix<Type>*, Matrix<Type>*);
// Eye matrix multiplication numerical checks
Matrix<Type>* EyeMatMulNum(Matrix<Type>*, Matrix<Type>*);


