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
bool IsZeroMatrix(const Matrix<Type> *);

// Zero matrix addition
const Matrix<Type> *ZeroMatAdd(const Matrix<Type> *, const Matrix<Type> *);
// Zero matrix scalar addition
const Matrix<Type> *ZeroMatScalarAdd(Type, const Matrix<Type> *);

// Zero matrix subtraction
const Matrix<Type> *ZeroMatSub(const Matrix<Type> *, const Matrix<Type> *);

// Zero matrix multiplication
const Matrix<Type> *ZeroMatMul(const Matrix<Type> *, const Matrix<Type> *);
// Zero matrix scalar multiplication
const Matrix<Type> *ZeroMatScalarMul(Type, const Matrix<Type> *);

// Zero matrix Kronocker product
const Matrix<Type> *ZeroMatKron(const Matrix<Type> *, const Matrix<Type> *);
// Zero matrix Hadamard product
const Matrix<Type> *ZeroMatHadamard(const Matrix<Type> *, const Matrix<Type> *);

// Zero matrix convolution
const Matrix<Type> *ZeroMatConv(const size_t, const size_t,
                                const Matrix<Type> *, const Matrix<Type> *);

// Zero matrix derivative convolution
const Matrix<Type> *ZeroMatDervConv(const size_t, const size_t,
                                    const Matrix<Type> *, const Matrix<Type> *,
                                    const Matrix<Type> *, const Matrix<Type> *);

// Zero matrix addition numerics
const Matrix<Type> *ZeroMatAddNum(const Matrix<Type> *, const Matrix<Type> *);
// Zero matrix scalar addition numerics
const Matrix<Type> *ZeroMatScalarAddNum(Type, const Matrix<Type> *);

// Zero matrix subtraction numerics
const Matrix<Type> *ZeroMatSubNum(const Matrix<Type> *, const Matrix<Type> *);

// Zero matrix multiplication numerics
const Matrix<Type> *ZeroMatMulNum(const Matrix<Type> *, const Matrix<Type> *);
// Zero matrix scalar multiplication numerics
const Matrix<Type> *ZeroMatScalarMulNum(Type, const Matrix<Type> *);

// Zero matrix kronocker product numerics
const Matrix<Type> *ZeroMatKronNum(const Matrix<Type> *, const Matrix<Type> *);
// Zero matrix Hadamard product numerics
const Matrix<Type> *ZeroMatHadamardNum(const Matrix<Type> *,
                                       const Matrix<Type> *);

// Zero matrix convolution numerics
const Matrix<Type> *ZeroMatConvNum(const size_t, const size_t,
                                   const Matrix<Type> *, const Matrix<Type> *);

// Zero matrix derivative convolution numerics
const Matrix<Type> *ZeroMatDervConvNum(const size_t, const size_t,
                                       const Matrix<Type> *,
                                       const Matrix<Type> *,
                                       const Matrix<Type> *,
                                       const Matrix<Type> *);


void SubZero(const Matrix<Type>*, Matrix<Type>*&);