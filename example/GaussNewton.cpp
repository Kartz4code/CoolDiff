/**
 * @file example/GaussNewton.cpp
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

// Number of data points
#define N 5

int main(int argc, char** argv) {

  // Create X-Y pairs of data of size N;
  Type* X = new Type[N]; 
  Type* Y = new Type[N];

  // Nonlinear function f(x) = 2*x*x + x + 1
  for(size_t i{}; i < N; ++i) {
    X[i] = i; Y[i] = 2*X[i]*X[i] + X[i] + 1;
  }
  
  // Parameters to store data 
  Parameter y, x;
  // Variables (coefficients)
  Variable b1{0}, b2{0}, b3{0}; 

  // Objective 
  Expression L = pow(y - (b1*x*x + b2*x + b3), 2);
  Oracle oracle(L,{b1,b2,b3});

  // Gauss-Newton Jacobian Matrix (#coefficients, data size)
  Matrix<Type> result(3,N);

  // Compute jacobians for individual data and store it in result matrix
  for(size_t i{}; i < N; ++i) {
    y = Y[i]; x = X[i];
    result.setBlockMat({0,2},{i,i}, &oracle.jacobian());  
  }

  std::cout << Eval(result) << "\n";
  
  /* Assuming after optimization step, after a change in coefficients */
  b1 = 4; b2 = 5; b3 = 6;

  // Compute jacobians for individual data and store it in result matrix   
  for(size_t i{}; i < N; ++i) {
    y = Y[i]; x = X[i];
    result.setBlockMat({0,2},{i,i}, &oracle.jacobian());  
  }

  std::cout << Eval(result) << "\n";

  delete[] X;
  delete[] Y;
  return 0;
}