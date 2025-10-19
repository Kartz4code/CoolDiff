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

class MetaMatrix : protected CommonHeader {
protected:
  // Vector of clones
  Vector<Matrix<Type>*> m_cloned{nullptr};
  // Counter values
  long long int m_clone_counter{};
  
  // Clear clone
  void clearClone();
  // Counter increment function   
  long long int incFunc(const size_t = 2);

public:
  // Visited flag
  bool m_visited{false};
  // Cache for reverse AD
  OMMatPair m_cache{};

  // Default constructor
  MetaMatrix() {
    const constexpr size_t init_size = 32;
    m_cloned.resize(init_size); 
    std::fill_n(EXECUTION_PAR m_cloned.begin(), init_size, nullptr);
  }

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
void ResetZero(Matrix<Type>*);
