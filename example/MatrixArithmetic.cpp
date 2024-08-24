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
  std::vector<Variable> X(10);
  std::fill(X.begin(), X.end(), 1.3);
  Expression y = sin(X[1])*X[0]/cos(X[2]*X[3])*X[4]*pow(X[5],2) + pow(2,X[6])/(X[7]*X[8]*X[9]);
  //y = y + 1;
  Expression y1 = y;

  std::cout <<  JacobR(y1, X) << "\n";
  std::cout <<  JacobF(y1, X) << "\n";

  std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();
  for(size_t i{}; i < 10000; ++i) {
    JacobF(y1, X, true);
  }
  std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
  std::cout << "Time difference (Serial) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - begin1).count() << "[ms]\n";

  std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
  for(size_t i{}; i < 10000; ++i) {
    JacobF(y1, X);
  }
  std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
  std::cout << "Time difference (Parallel) = " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - begin2).count() << "[ms]\n";
}

int main(int argc, char **argv) {
  //func();

  Matrix<Variable> x(2, 1);
  x(0, 0) = 2;
  x(1, 0) = 3;

  Matrix<Type> m = CreateMatrix<Type>(2, 2);
  m(0, 0) = 1;
  m(0, 1) = 2;
  m(1, 0) = 3;
  m(1, 1) = 4;

  Matrix<Expression> m1{2, 2};
  m1(0, 0) = x(0, 0) + x(1, 0);
  m1(0, 1) = x(0, 0) * x(1, 0);
  m1(1, 0) = x(0, 0) / x(1, 0);
  m1(1, 1) = x(0, 0) - x(1, 0);

  Matrix<Expression> sum = m1 + m;
  sum = sum + m;
  sum = m1 + sum;

  //std::cout << Eval(sum) << "\n";
  //std::cout << DevalF(sum, x(0, 0)) << "\n";

  x(0, 0) = 20;
  m1(1,1) = x(0,0) + x(0,0);

  //Eval(sum);
  std::cout << DevalF(sum,x) << "\n";

  x(0,0) = 5;

  std::cout << DevalF(sum,x) << "\n";

  return 0;
}