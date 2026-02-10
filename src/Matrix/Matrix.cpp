/**
 * @file src/Matrix/Matrix.cpp
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

#include "Matrix.hpp"
#include "MemoryManager.hpp"

namespace CoolDiff {
  namespace TensorR2 {
    namespace Details {
        Type ScalarSpl(const Matrix<Type>* mat) {
          Type value = *(mat->getMatrixPtr()); 
          return value;
      } 
    }
  }
}