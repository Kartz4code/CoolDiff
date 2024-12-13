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

#include "CoolDiff.hpp"
#include "MatOperators.hpp"

void func15() {
  Variable x{2};
  Expression y = x * x + 2 * x + 1;

  Expression yx = SymDiff(y, x);
  Expression yxx = SymDiff(yx, x);
  Expression z = SymDiff(x, x);

  std::cout << Eval(y) << "\n";
  std::cout << DevalF(y, x) << "\n";

  std::cout << Eval(yx) << "\n";
  std::cout << DevalF(yx, x) << "\n";

  std::cout << Eval(yxx) << "\n";
  std::cout << DevalF(yxx, x) << "\n";

  std::cout << Eval(z) << "\n";
  std::cout << DevalF(z, x) << "\n";
}

void func14() {
  Matrix<Type> A(3, 3);
  Matrix<Variable> X(3, 3);

  A(0, 0) = 1;
  A(0, 1) = 2;
  A(0, 2) = 3;
  A(1, 0) = 4;
  A(1, 1) = 5;
  A(1, 2) = 6;
  A(2, 0) = 7;
  A(2, 1) = 8;
  A(2, 2) = 9;

  X(0, 0) = 1;
  X(0, 1) = 2;
  X(0, 2) = 3;
  X(1, 0) = 4;
  X(1, 1) = 5;
  X(1, 2) = 6;
  X(2, 0) = 7;
  X(2, 1) = 8;
  X(2, 2) = 9;

  Matrix<Expression> exp = trace(A * X);
  std::cout << DevalF(exp, X) << "\n";
}

void func13() {
  Matrix<Type> X(3, 3);
  Matrix<Variable> W(2, 2), W2(2, 2);

  X(0, 0) = 1;
  X(0, 1) = 2;
  X(0, 2) = 3;
  X(1, 0) = 4;
  X(1, 1) = 4;
  X(1, 2) = 5;
  X(2, 0) = 6;
  X(2, 1) = 7;
  X(2, 2) = 8;

  W(0, 0) = 1;
  W(0, 1) = 2;
  W(1, 0) = 3;
  W(1, 1) = 4;

  W2(0, 0) = 3;
  W2(0, 1) = 4;
  W2(1, 0) = 5;
  W2(1, 1) = 6;

  Matrix<Expression> exp = conv(conv(X, W, 1, 1, 1, 1), W2, 1, 1, 1, 1);
  std::cout << DevalF(exp, W) << "\n";
  std::cout << DevalF(exp, W2) << "\n";
}

void func12() {
  Variable x1{2}, x2{5};
  std::cout << Eval(x1 * x2 + x1) << "\n";

  std::cout << DevalF(x1 * x2 + x1, x1) << "\n";
  std::cout << DevalR(x1, x1) << "\n";
}

void func11() {

  MatVariable &X = MatVariable::MatrixFactory::CreateMatrix(2, 3);
  X(0, 0) = 1;
  X(0, 1) = 2;
  X(0, 2) = 3;
  X(1, 0) = 4;
  X(1, 1) = 5;
  X(1, 2) = 6;

  Matrix<Type> A(3, 2);
  A(0, 0) = 4;
  A(0, 1) = 3;
  A(1, 0) = 2;
  A(1, 1) = 1;
  A(2, 0) = 37;
  A(2, 1) = 43;

  Matrix<Expression> Ym = A * X * A;
  std::cout << Eval(Ym) << "\n";

  Matrix<Expression> Y = sigma(transpose(X) * transpose(A));
  Y = Y * sigma(A * X);

  Matrix<Type> A1(2, 3);
  A1(0, 0) = 1;
  A1(0, 1) = 2;
  A1(0, 2) = 3;
  A1(1, 0) = 4;
  A1(1, 1) = 5;
  A1(1, 2) = 6;

  Matrix<Variable> X2(2, 4);
  X2(0, 0) = 1;
  X2(0, 1) = 2;
  X2(0, 2) = 3;
  X2(0, 3) = 7;
  X2(1, 0) = 4;
  X2(1, 1) = 5;
  X2(1, 2) = 6;
  X2(1, 3) = 8;

  Matrix<Expression> Y2 = X * A * X2;

  std::cout << Eval(Y2) << "\n";
  std::cout << Eval(Y2) << "\n";
  std::cout << DevalF(Y2, X) << "\n";
  std::cout << DevalF(Y2, X2) << "\n";

  std::cout << DevalF(Y2, X) << "\n";

  std::cout << Eval(Y) << "\n";
  std::cout << DevalF(Y, X) << "\n";
}


int main(int argc, char **argv) {
  func15();
  func14();
  func13();
  func12();
  func11();

  Matrix<Variable> x(2, 1);
  x(0, 0) = 2;
  x(1, 0) = 3;

  Matrix<Type> m = Matrix<Type>::MatrixFactory::CreateMatrix(2, 2);
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