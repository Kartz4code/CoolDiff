/**
 * @file example/MatrixArithmetic.cpp
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

#include "CommonMatFunctions.hpp"
#include "CoolDiff.hpp"

void func() {
  Variable x1{1}, x2{2};
  Expression y = x1 * x1 * x2 + x2 * x1;

  Expression y1 = y;

  std::cout << JacobF(y1, {x1, x2}) << "\n";
}

int main(int argc, char **argv) {
  func();

  Matrix<Variable> x(2, 1);
  x(0, 0) = 2;
  x(1, 0) = 3;

  Matrix<Type> m = CreateMatrix<Type>(3, 2);
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;

  Matrix<Expression> m1{3, 2};
  m1(0, 0) = x(0, 0) + x(1, 0);
  m1(0, 1) = x(0, 0) * x(1, 0);
  m1(1, 0) = x(0, 0) / x(1, 0);
  m1(1, 1) = x(0, 0) - x(1, 0);

  Matrix<Expression> sum = m1 + m;
  sum = sum + m;
  sum = m1 + sum;

  std::cout << Eval(sum) << "\n";
  std::cout << DevalF(sum, x(0, 0)) << "\n";

  x(0, 0) = 20;

  std::cout << Eval(sum) << "\n";
  std::cout << DevalF(sum, x(0, 0)) << "\n";

  return 0;
}