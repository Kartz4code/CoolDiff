/**
 * @file include/Matrix/MatOperators.hpp
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

// Matrix-Matrix addition - Left, Right, Result matrix pointer
void MatrixAdd(const Matrix<Type> *, const Matrix<Type> *, Matrix<Type> *&);

// Matrix-Matrix subtraction - Left, Right, Result matrix pointer
void MatrixSub(const Matrix<Type> *, const Matrix<Type> *, Matrix<Type> *&);

// Matrix-Matrix multiplication - Left, Right, Result matrix pointer
void MatrixMul(const Matrix<Type> *, const Matrix<Type> *, Matrix<Type> *&);

// Matrix-Matrix Kronocker product - Left, Right, Result matrix pointer
void MatrixKron(const Matrix<Type> *, const Matrix<Type> *, Matrix<Type> *&);

// Matrix-Matrix Hadamard product - Left, Right, Result matrix pointer
void MatrixHadamard(const Matrix<Type> *, const Matrix<Type> *,
                    Matrix<Type> *&);

// Matrix-Scalar addition
void MatrixScalarAdd(Type, const Matrix<Type> *, Matrix<Type> *&);

// Matrix-Scalar multiplication
void MatrixScalarMul(Type, const Matrix<Type> *, Matrix<Type> *&);

// Matrix transpose
void MatrixTranspose(const Matrix<Type> *, Matrix<Type> *&);

// Matrix derivative transpose
void MatrixDervTranspose(const size_t, const size_t, const size_t, const size_t,
                         const Matrix<Type> *, Matrix<Type> *&);

// Matrix convolution
void MatrixConv(const size_t, const size_t, const size_t, const size_t,
                const Matrix<Type> *, const Matrix<Type> *, Matrix<Type> *&);

// Matrix derivative convolution
void MatrixDervConv(const size_t, const size_t, const size_t, const size_t,
                    const size_t, const size_t, const Matrix<Type> *,
                    const Matrix<Type> *, const Matrix<Type> *,
                    const Matrix<Type> *, Matrix<Type> *&);
