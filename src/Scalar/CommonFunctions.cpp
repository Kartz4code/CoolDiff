/**
 * @file src/Scalar/CommonFunctions.cpp
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

#include "CommonFunctions.hpp"
#include "Matrix.hpp"

// Namespace CoolDiff
namespace CoolDiff {
  // Namespace Scalar
  namespace TensorR1 {
    // namespace Details
    namespace Details {
      // Evaluate function (Scalar)
      Type EvalExp(Expression& exp) {
        // Reset graph/tree
        exp.resetImpl();
        // Return evaluation value
        return exp.eval();
      }

      // Forward mode algorithmic differentiation (Scalar)
      Type DevalFExp(Expression& exp, const Variable& x) {
        // Reset visited flags in the tree
        exp.resetImpl();
        // Return forward differentiation value
        return exp.devalF(x);
      }

      // Reverse mode algorithmic differentiation (Scalar)
      Type DevalRExp(Expression& exp, const Variable& x) {
        // Return reverse differentiation value
        return exp.devalR(x);
      }

      // Main precomputation of reverse mode AD computation 1st
      void PreCompExp(Expression& exp) {
        // Reset visited flags in the tree
        exp.resetImpl();
        // Traverse tree
        exp.traverse();
      }

      // Main reverse mode AD table 1st
      OMPair& PreCompCacheExp(Expression& exp) {
        // Reset flags
        exp.resetImpl();
        // Traverse evaluation graph/tree
        exp.traverse();
        // Return cache
        return exp.getCache();
      }

      // Symbolic Expression
      Expression& SymDiffExp(Expression& exp, const Variable& var) {
        // Reset graph/tree
        return exp.SymDiff(var);
      }
    } 
  }
}