/**
 * @file include/Matrix/MatrixHandler/MatTranspose/NaiveCPU/MatDervTransposeNaiveHandler.hpp
 *
 * @copyright 2023-2026 Karthik Murali Madhavan Rathai
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

template<typename T, typename = std::enable_if_t<std::is_base_of_v<MatrixStaticHandler, T>>>
class MatDervTransposeNaiveHandler : public T {
    public:
      void handle(const size_t nrows_f, const size_t ncols_f, 
                  const size_t nrows_x, const size_t ncols_x, 
                  const Matrix<Type>* rhs, Matrix<Type>*& result) {
                    
        // Result matrix dimensions
        const size_t nrows{ncols_f * nrows_x};
        const size_t ncols{nrows_f * ncols_x};
      
        // Mat memory strategy
        const auto& rhs_strategy = rhs->allocatorType();

        /* Matrix derivative transpose */
        // Pool matrix
        MemoryManager::MatrixPool(result, nrows, ncols, rhs_strategy);
      
        const auto outer_idx = CoolDiff::Common::Range<size_t>(0, ncols_f * nrows_f);
        const auto inner_idx = CoolDiff::Common::Range<size_t>(0, ncols_x * nrows_x);
      
        // Outer loop
        std::for_each(EXECUTION_PAR outer_idx.begin(), outer_idx.end(), [&](const size_t n1) {
              // Outer Row and column index
              const size_t j = (n1 % ncols_f);
              const size_t i = ((n1 - j) / ncols_f);
              // Inner loop
              std::for_each(EXECUTION_PAR inner_idx.begin(), inner_idx.end(),
                            [&](const size_t n2) {
                              // Inner Row and column index
                              const size_t m = (n2 % ncols_x);
                              const size_t l = ((n2 - m) / ncols_x);
                              #if defined(USE_COMPLEX_MATH)
                                (*result)(l + (j * nrows_x), m + (i * ncols_x)) = std::conj((*rhs)(l + (i * nrows_x), m + (j * ncols_x)));
                              #else
                                (*result)(l + (j * nrows_x), m + (i * ncols_x)) = (*rhs)(l + (i * nrows_x), m + (j * ncols_x));
                              #endif
              });
        });
      }
};