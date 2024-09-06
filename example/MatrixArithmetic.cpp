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
#include "MatOperators.hpp"
#include "MatrixBasics.hpp"

void func9() {
  Matrix<Variable> x(2, 2);
  x(0, 0) = 1;
  x(0, 1) = 5;
  x(1, 0) = 3;
  x(1, 1) = 20;

  Matrix<Type> x2(2, 2);
  x2(0, 0) = 1;
  x2(0, 1) = 5;
  x2(1, 0) = 3;
  x2(1, 1) = 20;

  Matrix<Expression> S = x - x + x;

  std::cout << Eval(S) << "\n";
  std::cout << DevalF(S, x) << "\n";
}

void func2() {
  Matrix<Variable> x(2, 2);
  x(0, 0) = 1;
  x(0, 1) = 5;
  x(1, 0) = 3;
  x(1, 1) = 20;

  Matrix<Type> m = CreateMatrix<Type>(2, 2);
  m(0, 0) = 1;
  m(0, 1) = 0;
  m(1, 0) = 0;
  m(1, 1) = 1;

  Matrix<Expression> x2(2, 2);
  x2(0, 0) = x(0, 0) + x(1, 1);
  x2(0, 1) = x(0, 0) - x(1, 1);
  x2(1, 0) = x(1, 0) + x(1, 0);
  x2(1, 1) = x(0, 0) + x(1, 1) + x(0, 1) + x(1, 0);

  Matrix<Expression> sum = m * x2;
  sum = sum * x + x;

  std::cout << Eval(sum) << "\n";
  std::cout << DevalF(sum, x) << "\n";
  std::cout << DevalF(sum, x) << "\n";
}

void func3() {
  Matrix<Variable> x(2, 2);
  x(0, 0) = 1;
  x(0, 1) = 2;
  x(1, 0) = 3;
  x(1, 1) = 4;

  Matrix<Type> m = CreateMatrix<Type>(2, 2);
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;

  Matrix<Expression> x2(2, 2);
  x2(0, 0) = x(0, 0) + x(1, 0);
  x2(0, 1) = x(1, 1) + x(0, 1);
  x2(1, 0) = x(1, 0) + x(1, 1);
  x2(1, 1) = x(1, 1);

  Matrix<Type> m2(2, 1);
  m2(0, 0) = 1;
  m2(1, 0) = 5;

  Matrix<Expression> sum = x2 * x2 * x2 + x;
  sum = sum * m + x;
  sum = x * sum * m2;

  std::cout << DevalF(sum, x) << "\n";
  std::cout << DevalF(sum, x) << "\n\n";
}

void func5() {
  auto m1 = CreateMatrix<Variable>(2, 1);
  m1(0, 0) = 1;
  m1(1, 0) = 1;

  Matrix<Type> m2 = CreateMatrix<Type>(2, 2);
  m2(0, 0) = 1;
  m2(0, 1) = 2;
  m2(1, 0) = 3;
  m2(1, 1) = 4;

  Matrix<Expression> M = m1 + m1;
  M = m2 * M + m1;

  std::cout << Eval(M) << "\n";
  std::cout << DevalF(M, m1) << "\n";
  std::cout << DevalF(M, m1) << "\n";
}

int main(int argc, char **argv) {
  func9();
  func2();
  func3();
  func5();

  Matrix<Variable> x(2, 1);
  x(0, 0) = 2;
  x(1, 0) = 3;

  Matrix<Type> m = CreateMatrix<Type>(2, 2);
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;

  Matrix<Type> m2(2, 1);
  m2(0, 0) = 1;
  m2(1, 0) = 3;

  Matrix<Expression> m1{2, 2};

  Expression y = x(0, 0) + x(1, 0);

  m1(0, 0) = x(0, 0) + x(1, 0);
  m1(0, 1) = x(1, 0);
  m1(1, 0) = x(1, 0);
  m1(1, 1) = x(0, 0) + x(1, 0);

  Matrix<Expression> sum = m * x;
  sum = m * sum + x + x;

  std::cout << Eval(sum) << "\n";
  std::cout << DevalF(sum, x) << "\n";

  x(0, 0) = 5;

  std::cout << Eval(sum) << "\n";
  std::cout << DevalF(sum, x) << "\n";
  std::cout << DevalF(sum, x) << "\n";

  return 0;
}