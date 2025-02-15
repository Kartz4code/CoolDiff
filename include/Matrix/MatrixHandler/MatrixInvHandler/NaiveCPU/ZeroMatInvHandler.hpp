/**
 * @file include/Matrix/MatrixHandler/MatrixInvHandler/NaiveCPU/ZeroMatInvHandler.hpp
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

 #include "MatrixStaticHandler.hpp"
 #include "Matrix.hpp"
 #include "MatrixZeroOps.hpp"

 template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
 class ZeroMatInvHandler : public T {
    public:
    void handle(const Matrix<Type>* mat, Matrix<Type>*& result) {
      #if defined(NAIVE_IMPL)
        /* Zero matrix special check */
        if (true == IsZeroMatrix(mat)) {
          ASSERT(false, "Inverse of a zero matrix");
          return;
        }
      #endif
        // Chain of responsibility
        T::handle(mat, result);
      }
 };