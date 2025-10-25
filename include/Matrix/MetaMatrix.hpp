/**
 * @file include/Matrix/MetaMatrix.hpp
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

// MetaMatrix class
class MetaMatrix : protected CommonHeader {
  public:  
    // Long long type
    using LL_t = long long int;

  private:
    const size_t m_init_size = 16;

  protected:
  // Vector of clones
  Vector<Matrix<Type>*> m_cloned{nullptr};
  // Counter values
  LL_t m_clone_counter{};
  
  // Clear clone
  void clearClone();
  // Counter increment function   
  LL_t incFunc(const size_t = 2);

public:
  // Visited flag
  bool m_visited{false};
  // Cache for reverse AD
  OMMatPair m_cache{};

  // Default constructor
  MetaMatrix();

  // Evaluate run-time
  V_PURE(Matrix<Type>* eval());

  // Forward derivative
  V_PURE(Matrix<Type>* devalF(Matrix<Variable>&));

  V_UNPURE(void traverse(OMMatPair* = nullptr)) {
    ASSERT(false, "An operation involved in the matrix expression is not yet implemented");
  }

  V_UNPURE(OMMatPair& getCache()) {
    return m_cache;
  }

  // Get number of rows and columns
  V_PURE(size_t getNumRows() const);

  V_PURE(size_t getNumColumns() const);

  // Reset all visited flags
  V_PURE(void reset());

  // Get type
  V_PURE(std::string_view getType() const);

  // Destructor
  V_DTR(~MetaMatrix()) = default;
};

// Reset type matrix (Set the values of MatType to zero)
namespace CoolDiff {
  namespace TensorR2 {
    namespace Details {
      void ResetZero(Matrix<Type>*);
    }
  }
}
