/**
 * @file src/Matrix/MatrixHandler/ZeroMatAddHandler.cpp
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

#include "ZeroMatAddHandler.hpp"
#include "MatrixZeroOps.hpp"
#include "Matrix.hpp"

void ZeroMatAddHandler::handle(Matrix<Type>* lhs, 
                              Matrix<Type>* rhs, 
                              Matrix<Type>*& result) {
  // Null pointer check
  NULL_CHECK(lhs, "LHS Matrix (lhs) is a nullptr");
  NULL_CHECK(rhs, "RHS Matrix (rhs) is a nullptr");

  /* Zero matrix special check */
  if(auto* it = ZeroMatAdd(lhs, rhs); nullptr != it) {
      result = it;
      return;
  }  
  /* Zero matrix numerical check */
  else if(auto* it = ZeroMatAddNum(lhs, rhs); nullptr != it) {   
      result = it;
      return;
  }

  // Chain of responsibility 
  MatrixHandler::handle(lhs, rhs, result);
}