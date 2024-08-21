/**
 * @file example/Gaussian.cpp
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

#include "CoolDiff.hpp"

#define PI 3.14159

// Gaussian univariate distribution function
Expression &Gaussian(const Variable &x, Type mu, Type sig) {
  auto &gauss = CreateExpr(0);
  gauss = (1 / sqrt(2 * PI * pow(sig, 2))) *
          exp(-pow((x - mu), 2) / (2 * pow(sig, 2)));
  return gauss;
}

int main(int argc, char **argv) {
  Variable x{10.34};
  Expression gauss = Gaussian(x, 1, 0.9);

  // Evaluate Gaussian function
  std::cout << "Evaluation: " << Eval(gauss) << "\n";
  // Forward derivative
  std::cout << "Forward derivative: " << DevalF(gauss, x) << "\n";
  // Precompute the adjoints
  PreComp(gauss);
  // Reverse derivative
  std::cout << "Reverse derivative: " << DevalR(gauss, x) << "\n\n";

  // Symbolic differentiation
  auto &gauss_derv = SymDiff(gauss, x);

  // Evaluate Gaussian function on symbolic derivative
  std::cout << "Evaluation(Sym): " << Eval(gauss_derv) << "\n";
  // Forward derivative
  std::cout << "Forward derivative(Sym): " << DevalF(gauss_derv, x) << "\n";
  // Precompute the adjoints
  PreComp(gauss_derv);
  // Reverse derivative
  std::cout << "Reverse derivative(Sym): " << DevalR(gauss_derv, x) << "\n";

  return 0;
}