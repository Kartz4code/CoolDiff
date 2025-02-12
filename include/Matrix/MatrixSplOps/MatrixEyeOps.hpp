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
bool IsEyeMatrix(const Matrix<Type> *);

// Eye matrix addition
const Matrix<Type> *EyeMatAdd(const Matrix<Type> *, const Matrix<Type> *);
// Eye matrix scalar addition
const Matrix<Type> *EyeMatScalarAdd(Type, const Matrix<Type> *);

// Eye matrix subtraction
const Matrix<Type> *EyeMatSub(const Matrix<Type> *, const Matrix<Type> *);

// Eye matrix multiplication
const Matrix<Type> *EyeMatMul(const Matrix<Type> *, const Matrix<Type> *);
// Eye matrix scalar addition
const Matrix<Type> *EyeMatScalarMul(Type, const Matrix<Type> *);

// Eye matrix Kronocker product
const Matrix<Type> *EyeMatKron(const Matrix<Type> *, const Matrix<Type> *);
// Eye matrix Hadamard product
const Matrix<Type> *EyeMatHadamard(const Matrix<Type> *, const Matrix<Type> *);

// Eye matrix addition numerics
const Matrix<Type> *EyeMatAddNum(const Matrix<Type> *, const Matrix<Type> *);
// Eye matrix scalar addition
const Matrix<Type> *EyeMatScalarAddNum(Type, const Matrix<Type> *);

// Eye matrix subtraction numerics
const Matrix<Type> *EyeMatSubNum(const Matrix<Type> *, const Matrix<Type> *);

// Eye matrix multiplication numerics
const Matrix<Type> *EyeMatMulNum(const Matrix<Type> *, const Matrix<Type> *);
// Zero matrix scalar multiplication numerics
const Matrix<Type> *EyeMatScalarMulNum(Type, const Matrix<Type> *);

// Eye matrix Kronocker product numerics
const Matrix<Type> *EyeMatKronNum(const Matrix<Type> *, const Matrix<Type> *);
// Eye matrix Hadamard product numerics
const Matrix<Type> *EyeMatHadamardNum(const Matrix<Type> *,
                                      const Matrix<Type> *);

namespace BaselineCPU {
    void AddEye(const Matrix<Type>*, Matrix<Type>*&);
    void Add2Eye(const Matrix<Type>*, Matrix<Type>*&);
    void AddEye(Type, const Matrix<Type>*, Matrix<Type>*&);

    void MulEye(Type, const Matrix<Type>*, Matrix<Type>*&);

    void SubEyeRHS(const Matrix<Type>*, Matrix<Type>*&);
    void SubEyeLHS(const Matrix<Type>*, Matrix<Type>*&);

    void KronEyeLHS(const Matrix<Type>*, const Matrix<Type>*, Matrix<Type>*&);
    void KronEyeRHS(const Matrix<Type>*, const Matrix<Type>*, Matrix<Type>*&);

    void HadamardEye(const Matrix<Type>*, Matrix<Type>*&);
}