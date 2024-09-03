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

#include "Matrix.hpp"
#include "GenericMatSum.hpp"
#include "GenericMatProduct.hpp"


/*
// Is the matrix ones?
bool IsOnesMatrix(Matrix<Type>*);

// Is the matrix diagonal?
bool IsDiagMatrix(Matrix<Type>*);

// Is the row matrix ?
bool IsRowMatrix(Matrix<Type>*);

// Is the column matrix ?
bool IsColMatrix(Matrix<Type>*);

// Find type of matrix
size_t FindMatType(const Matrix<Type>&);
*/

// Matrix evaluation
Matrix<Type> &Eval(Matrix<Expression>&);

// Matrix-Matrix derivative evaluation
Matrix<Type>& DevalF(Matrix<Expression>&, Matrix<Variable>&);

