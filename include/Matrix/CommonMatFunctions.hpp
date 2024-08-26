/**
 * @file include/Matrix/CommonMatFunctions.hpp
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

#include "GenericMatSum.hpp"
#include "GenericMatProduct.hpp"
#include "Matrix.hpp"


// Is the matrix zero?
bool IsZeroMatrix(const Matrix<Type> &);

// Is the matrix identity?
bool IsEyeMatrix(const Matrix<Type> &);

// Is the matrix ones?
bool IsOnesMatrix(const Matrix<Type> &);

// Is the matrix square?
bool IsSquareMatrix(const Matrix<Type> &);

// Is the matrix diagonal?
bool IsDiagMatrix(const Matrix<Type> &);

// Is the row matrix ?
bool IsRowMatrix(const Matrix<Type>&);

// Is the column matrix ?
bool IsColMatrix(const Matrix<Type>&);

// Find type of matrix
size_t FindMatType(const Matrix<Type>&);

// Matrix evaluation
Matrix<Type> &Eval(Matrix<Expression>&);

// Matrix derivative evaluation
Matrix<Type> &DevalF(Matrix<Expression>&, const Variable &);

// Matrix-Matrix derivative evaluation
Matrix<Type>& DevalF(Matrix<Expression>&, const Matrix<Variable>&);

