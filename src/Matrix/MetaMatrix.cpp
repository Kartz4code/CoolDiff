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
namespace CoolDiff {
  namespace TensorR2 {
    namespace Details {
      void ResetZero(Matrix<Type>* ptr) {
        if ((nullptr != ptr) && (-1 == ptr->getMatType())) {
          const size_t size = ptr->getNumElem();
          auto* mptr = ptr->getMatrixPtr();
          std::fill(EXECUTION_PAR mptr, mptr + size, (Type)(0));
        }
      }
    }
  }
}

// Default constructor
MetaMatrix::MetaMatrix() {
  m_cloned.resize(m_init_size); 
  std::fill_n(EXECUTION_PAR m_cloned.begin(), m_init_size, nullptr);
}

MetaMatrix::LL_t MetaMatrix::incFunc(const size_t scale) {
  if(const auto size = (LL_t)m_cloned.size(); (LL_t)m_clone_counter >= (size-1)) {
      m_cloned.resize(scale*size);
  }
  m_clone_counter += 1;
  return m_clone_counter;
}

// Clear clone counter
void MetaMatrix::clearClone() {
  m_clone_counter = -1;
}
