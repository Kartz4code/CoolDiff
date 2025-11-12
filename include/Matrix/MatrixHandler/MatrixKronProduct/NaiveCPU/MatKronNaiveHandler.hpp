/**
 * @file include/Matrix/MatrixHandler/MatrixKronProduct/NaiveCPU/MatKronNaiveHandler.hpp
 *
 * @copyright 2023-2025 Karthik Murali Madhavan Rathai
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
class MatKronNaiveHandler : public T {
  public:
  void handle(const Matrix<Type>* lhs, const Matrix<Type>* rhs, Matrix<Type>*& result) {
      /* Matrix-Matrix numerical Kronocker product */

      // Dimensions of LHS and RHS matrices
      const size_t lr{lhs->getNumRows()};
      const size_t lc{lhs->getNumColumns()};
      const size_t rr{rhs->getNumRows()};
      const size_t rc{rhs->getNumColumns()};

      // Pool matrix
      MemoryManager::MatrixPool((lr * rr), (lc * rc), result);

      const auto lhs_idx = CoolDiff::Common::Range<size_t>(0, (lr * lc));
      const auto rhs_idx = CoolDiff::Common::Range<size_t>(0, (rr * rc));

      std::for_each(EXECUTION_PAR lhs_idx.begin(), lhs_idx.end(), [&](const size_t n1) {
        const size_t j = (n1 % lc);
        const size_t i = ((n1 - j) / lc);
        Type val = (*lhs)(i, j);

        if ((Type)(0) != val) {
          std::for_each(EXECUTION_PAR rhs_idx.begin(), rhs_idx.end(),
              [&](const size_t n2) {
                const size_t m = (n2 % rc);
                const size_t l = ((n2 - m) / rc);
                (*result)((i * rr) + l, (j * rc) + m) = ((*rhs)(l, m) * val);
              });
        }
      });

      return;
  }
};