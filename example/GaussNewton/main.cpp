/**
 * @file example/GaussNewton/GaussNewton.cpp
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

#include "GaussNewton.hpp"

// Number of data points
#define N (5)

Type g_input[N][2] = { {5.7548, 3.0242},
                       {3.9920, 5.0718},
                       {1.9450, 3.0839},
                       {4.0819, 0.9883},
                       {6.0082, 3.0074}};

Type g_output[N][2] = { {10.5143, -6.3945},
                        {11.3523, -3.7623},
                        {8.5288, -3.1747},
                        {7.8168, -6.1075},
                        {10.4434, -6.5830}};

// 2D ICP
void GNMatrix() {
  // Create X-Y pairs of data of size N;
  Matrix<Type> X(N,2); 
  Matrix<Type> Y(N,2);

  // Set up data
  for(int i{}; i < N; ++i) {
    for(int j{}; j < 2; ++j) {
      X(i,j) = g_input[i][j];
      Y(i,j) = g_output[i][j];
    }
  }

  Matrix<Variable> V(1,3);
  V[0] = 0; V[1] = 20; V[2] = -10;

  Matrix<Expression> R(2,2);
  Matrix<Expression> t(2,1); 

  R(0,0) = cos( V[0] ); R(0,1) = sin( V[0] );
  R(1,0) = -sin( V[0] ); R(1,1) = cos( V[0] );
  t[0] = V[1]; t[1] = V[2]; 

  Matrix<Parameter> PX(2,1);
  Matrix<Parameter> PY(2,1);

  Matrix<Expression> J = (PY - (R*PX + t));
  Oracle* oracle = Oracle::OracleFactory::CreateOracle(J,V);

  GaussNewton gn;
  gn.setData(&X,&Y,N)
    .setOracle(oracle)
    .setParameters(&PX,&PY)
    .setMaxIterations(4);

  TIME_IT_MS(gn.solve());
  TIME_IT_MS(gn.solve());

  std::cout << "Finally: " << Eval(V) << "\n";

}

int main(int argc, char** argv) {
  GNMatrix();
  return 0;
}