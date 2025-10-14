/**
 * @file include/Matrix/MatrixBasics.hpp
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

namespace CoolDiff {
    namespace TensorR2 {
        namespace MatrixBasics {
            /* Pointer semantics */
            // Numerical Eye matrix
            const Matrix<Type>* Eye(const size_t);

            // Numerical Zeros matrix
            const Matrix<Type>* Zeros(const size_t, const size_t);

            // Numerical Zeros square matrix
            const Matrix<Type>* Zeros(const size_t);

            // Numerical Ones matrix
            const Matrix<Type>* Ones(const size_t, const size_t);

            // Numerical Ones square matrix
            const Matrix<Type>* Ones(const size_t);

            // References
            // Numerical Eye matrix
            const Matrix<Type>& EyeRef(const size_t);

            // Numerical Ones matrix
            const Matrix<Type>& OnesRef(const size_t, const size_t);

            // Numerical Zero matrix
            const Matrix<Type>& ZerosRef(const size_t, const size_t);
        }
    }
}
