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
#define N (500)

// 2D ICP

void GNMatrix() {
  // Create X-Y pairs of data of size N;
  Matrix<Type> X(N,2); 
  Matrix<Type> Y(N,2);

  // Set up data
  for(int i{}; i < N; ++i) {
    for(int j{}; j < 2; ++j) {
      Type x = i + 0.00000000001; Type y = j + 0.00000000001;
      X(i,j) = x*y;
      Y(i,j) = y*x + 1;
    }
  }

  Matrix<Variable> V(1,3);
  V[0] = 3.14156/2; V[1] = 1; V[2] = 2;

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
    .setParameters(&PX,&PY);

  for(int i{}; i < 10; ++i) {
    V[0] = i+0.00000000001; V[1] = 4*i; V[2] = 5*i;
    Pair<MatType*, MatType*> res;
    TIME_IT_MS(res = gn.getAB());
    std::cout << *(res.first) << "\n";
    std::cout << *(res.second) << "\n";
  }
}

// 2D ICP
void GNScalar() {
  // Create X-Y pairs of data of size N;
  Matrix<Type> X(N,2); 
  Matrix<Type> Y(N,2);

  // Set up data
  for(int i{}; i < N; ++i) {
    for(int j{}; j < 2; ++j) {
      Type x = i + 0.00000000001; Type y = j + 0.00000000001;
      X(i,j) = x*y;
      Y(i,j) = y*x + 1;
    }
  }

  Matrix<Variable> V(3,1);
  V[0] = 3.14156/2; V[1] = 1; V[2] = 2;

  Matrix<Expression> R(2,2);
  Matrix<Expression> t(2,1); 

  R(0,0) = cos( V[0] ); R(0,1) = sin( V[0] );
  R(1,0) = -sin( V[0] ); R(1,1) = cos( V[0] );
  t[0] = V[1]; t[1] = V[2]; 

  Matrix<Parameter> PX(2,1);
  Matrix<Parameter> PY(2,1);

  Matrix<Expression> RPX(2,1);
  Expression J;

  for(size_t i{}; i < 2; ++i) {
    for(size_t j{}; j < 2; ++j) {
        RPX(i,0) = RPX(i,0) + R(i,j)*PX(j,0); 
    }
    J = J + (PY(i,0) - (RPX(i,0) + t(i,0)));
  } 

  //Matrix<Expression> J = sigma(PY - (RPX + t));
  Oracle* oracle = Oracle::OracleFactory::CreateOracle(J,V);

  GaussNewton gn;
  gn.setData(&X,&Y,N)
    .setOracle(oracle)
    .setParameters(&PX,&PY);

  for(int i{}; i < 10; ++i) {
    V[0] = i; V[1] = 4*i; V[2] = 5*i;
    Pair<MatType*, MatType*> res;
    TIME_IT_MS(res = gn.getAB());
    std::cout << *(res.first) << "\n";
    std::cout << *(res.second) << "\n";
    
  }
}

int main(int argc, char** argv) {
  GNMatrix();
  //GNScalar();
  return 0;
}