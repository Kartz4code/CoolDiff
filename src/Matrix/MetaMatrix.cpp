/**
 * @file src/Matrix/MetaMatrix.cpp
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

#include "MetaMatrix.hpp"
#include "Matrix.hpp"

// Reset type matrix (Set the values of MatType to zero)
void ResetZero(Matrix<Type> *ptr) {
  if ((nullptr != ptr) && (-1 == ptr->getMatType())) {
    const size_t size = ptr->getNumElem();
    auto *mptr = ptr->getMatrixPtr();
    std::fill(EXECUTION_PAR mptr, mptr + size, (Type)(0));
  }
}