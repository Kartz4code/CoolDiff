/**
 * @file include/Matrix/MatrixHandler/MatrixUnaryHandler/NaiveCPU/EyeMatUnaryHandler.hpp
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
#include "MatrixEyeOps.hpp"

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class EyeMatUnaryHandler : public T {
    public:
      void handle(const Matrix<Type>* mat, const FunctionType1& func, Matrix<Type>*& result) {
        #if defined(USE_SYMBOLIC_CHECK)
          /* Zero matrix special check */
          if (true == IsEyeMatrix(mat)) {
            // Rows and columns of result matrix
            const size_t nrows{mat->getNumRows()};
            const size_t ncols{mat->getNumColumns()};

            // Result matrix is transposed zero matrix
            MemoryManager::MatrixPool(nrows, ncols, result);

            // Eye matrix
            const auto idx = Range<size_t>(0, (nrows * ncols));
            std::for_each(EXECUTION_PAR idx.begin(), idx.end(), [&](const size_t n) {
              const size_t j = (n % ncols);
              const size_t i = ((n - j) / ncols);
              (*result)(i, j) = ((i == j) ? func((Type)1) : func((Type)0));
            });
            return;
          }
        #endif
          // Chain of responsibility
          T::handle(mat, func, result);
      } 
};