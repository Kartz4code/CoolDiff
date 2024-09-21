/**
 * @file include/Matrix/CommonMatFunctions.hpp
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

#include "Matrix.hpp"
#include "MatrixBasics.hpp"
#include "GenericMatHadamard.hpp"
#include "GenericMatProduct.hpp"
#include "GenericMatSub.hpp"
#include "GenericMatSum.hpp"
#include "GenericMatTranspose.hpp"
#include "GenericMatSigma.hpp"

// Matrix evaluation
template<typename T>
Matrix<Type> &Eval(Matrix<T> &Mexp) {
  // Reset graph/tree
  Mexp.resetImpl();
  // Return evaluation value
  return *(Mexp.eval());
}

// Matrix-Matrix derivative evaluation
template<typename T>
Matrix<Type> &DevalF(Matrix<T> &Mexp, Matrix<Variable> &X) {
  // Reset graph/tree
  Mexp.resetImpl();
  // Return evaluation value
  return *(Mexp.devalF(X));
}