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
#include "GaussNewtonData.hpp"
#include <fstream>

// Count number of rows
int CountRows(std::string_view file) {
  if (auto ifs = std::ifstream{file.data()}; ifs.is_open() == true) {
    return std::count(std::istreambuf_iterator<char>(ifs),
                      std::istreambuf_iterator<char>(), '\n');
  } else {
    return 0;
  }
}

// Counter number of cols
int CountCols(std::string_view file) {
  if (auto ifs = std::ifstream{file.data()}; ifs.is_open() == true) {
    std::string str{};
    std::getline(ifs, str);
    std::istringstream iss{str};
    Type value{};
    int cols{};
    while (iss >> value) {
      ++cols;
    }
    return cols;
  } else {
    return 0;
  }
}

// Get an input/output paired dataset
Pair<Matrix<Type> *, Matrix<Type> *> LoadData() {
  std::string_view in_file = GaussNewtonData::g_input_data;
  std::string_view out_file = GaussNewtonData::g_output_data;
  std::string str{};

  // Check for input/output matrix consistency
  int r_in = CountRows(in_file);
  int r_out = CountRows(out_file);
  int c_in = CountCols(in_file);
  int c_out = CountCols(out_file);

  ASSERT((r_in == r_out), "Matrix input/output rows not matching");
  ASSERT((c_in == c_out), "Matrix input/output columns not matching");

  // Pointer to input/output data
  Matrix<Type> *X = Matrix<Type>::MatrixFactory::CreateMatrixPtr(r_in, c_in);
  Matrix<Type> *Y = Matrix<Type>::MatrixFactory::CreateMatrixPtr(r_out, c_out);

  // Input file streams
  int i{};
  if (auto ifs = std::ifstream{in_file.data()}; ifs.is_open() == true) {
    while (std::getline(ifs, str)) {
      std::istringstream iss{str};
      for (size_t j{}; j < c_in; ++j) {
        iss >> (*X)(i, j);
      }
      ++i;
    }
  }

  // Output file streams
  i = 0;
  if (auto ifs = std::ifstream{out_file.data()}; ifs.is_open() == true) {
    while (std::getline(ifs, str)) {
      std::istringstream iss{str};
      for (size_t j{}; j < c_out; ++j) {
        iss >> (*Y)(i, j);
      }
      ++i;
    }
  }

  return {X, Y};
}


void RModeDerv() {
  Matrix<Variable> X(2,2);
  X(0,0) = 4;  X(0,1) = 3; 
  X(1,0) = 2;  X(1,1) = 1;

  Matrix<Variable> W1(2,1);
  Matrix<Variable> W2(2,1);

  // Set W1
  for(int i{}; i < 2; i++) {
    for(int j{}; j < 1; j++) {
      W1(i,j) = i+j+1;
    }
  }
  // Set W2
  for(int i{}; i < 2; i++) {
    for(int j{}; j < 1; j++) {
      W2(i,j) = i+2*j+2;
    }
  }

  auto doubler = X*X;
  auto smax = SoftMax(doubler);
  Matrix<Expression> l3 = trace(SinM((transpose(X)^X)*inv(X)*smax + 6.2*X)*X)*2;



  Matrix<Expression> res = -1.25*CosM(l3);
  res = 10 + res*CosM(res*res) + 2.24*res + trace(X)*det(X);
  res = res + det(X) + trace(X) + l3*MatrixFrobeniusNorm(doubler*X) + SinM(res - res*res);
  res = res + res + Sigma(doubler);

  Matrix<Expression> tester = doubler;

  /*  
  auto X1 = X;
  auto X2 = X1*X;
  Matrix<Expression> res = MatrixFrobeniusNorm(X2*X2);
  res = res + trace(X1*X2) + trace(X1*X2);
  */

  res.resetImpl();
  res.traverse();

  //auto it1 = res.getCache()[W1.m_nidx];
  //auto it2 = res.getCache()[W2.m_nidx];
  auto it3 = res.getCache()[X.m_nidx];
  
  //std::cout << Eval(*it1) << "\n";
  //std::cout << Eval(*it2) << "\n";
  std::cout << CoolDiff::Tensor2R::Eval(*it3) << "\n";
  std::cout << CoolDiff::Tensor2R::Eval(tester) << "\n";

  X[0] = 1;  X[1] = 2; 
  X[2] = 3;  X[3] = 4;

  res.resetImpl();
  res.traverse();

  //it1 = res.getCache()[W1.m_nidx];
  //it2 = res.getCache()[W2.m_nidx];
  it3 = res.getCache()[X.m_nidx];
  
  //std::cout << Eval(*it1) << "\n";
  //std::cout << Eval(*it2) << "\n";
  std::cout << CoolDiff::Tensor2R::Eval(*it3) << "\n";
  std::cout << CoolDiff::Tensor2R::Eval(tester) << "\n";
}


// 2D data matching
void GNMatrix() {
  // Load data
  auto data = LoadData();

  // Matrix variables
  Matrix<Variable> V(1, 3);
  V[0] = 0.5142;
  V[1] = 20;
  V[2] = -20;

  // Translation matrices
  Matrix<Variable> t(2,1,V.getMatrixPtr()+1);
  
  // Rotation matrix
  Matrix<Expression> R(2, 2);
  R(0, 0) = cos(V[0]); R(0, 1) = sin(V[0]);
  R(1, 0) = -sin(V[0]); R(1, 1) = cos(V[0]);


  // Parameter for input/output data
  Matrix<Parameter> PX(2, 1), PY(2, 1);

  // Matrix expression error function
  Matrix<Expression> J = (PY - (R * PX + t));

  GaussNewton gn;
  gn.setData(data.first, data.second)
    .setOracle(Oracle::OracleFactory::CreateOracle(J, V))
    .setDataParameters(&PX, &PY)
    .setMaxIterations(3);

  TIME_IT_MS(gn.solve());

  std::cout << "Computed values: " << CoolDiff::Tensor2R::Eval(V);
  std::cout << "Actual values: " << (Type)3.14159 / 2 << " " << (Type)5 << " " << (Type)-2 << "\n";
}

void NonLinearSolve() {
  Variable x{10}, y{15};

  Matrix<Variable> X(1, 2);
  X(0, 0) = x;
  X(0, 1) = y;
  Matrix<Expression> E(2, 1);

  E(0, 0) = x * x + y * y - 20;
  E(1, 0) = x - y + 2;

  GaussNewton gn;
  gn.setOracle(Oracle::OracleFactory::CreateOracle(E, X))
    .setMaxIterations(3);

  TIME_IT_US(gn.solve());
  TIME_IT_US(gn.solve());

  std::cout << "Computed values (x,y): " << CoolDiff::TensorR1::Eval(x) << "," << CoolDiff::TensorR1::Eval(y) << "\n";
  std::cout << "Actual values (x,y): (-4,-2) or (2,4)\n\n";
}

void ScalarSolve() {
  Variable x{10};

  Expression E = x*x - 2*x + 1;
  Matrix<Variable> X(1, 1);
  X(0,0) = x;

  GaussNewton gn;
  gn.setOracle(Oracle::OracleFactory::CreateOracle(E, X))
    .setMaxIterations(10);

  TIME_IT_US(gn.solve());

  std::cout << "Computed values: " << CoolDiff::TensorR1::Eval(x) << "\n";
}


int main(int argc, char **argv) { 
  GNMatrix();
  NonLinearSolve();
  RModeDerv();
  //RModeDerv();
  return 0;
}