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
#include <random>
// Create a random device and seed the engine
std::random_device rd;
std::mt19937 gen(rd()); // Mersenne Twister engine
// Define the range for your floating point numbers
std::uniform_real_distribution<> dis(0, +0.088); // Range: [1.0, 10.0)


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
Pair<Matrix<Type>*, Matrix<Type>*> LoadData() {
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

  std::cout << "Computed values: " << CoolDiff::TensorR2::Eval(V);
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

void FillRandomWeights(MatType& M) {
  for(int i{}; i < M.getNumRows(); ++i) {
    for(int j{}; j < M.getNumColumns(); ++j) {
      M(i,j) = dis(gen);
    }
  }
}

void GDOptimizer(Matrix<Type>& X, Matrix<Type>& dX, const Type& alpha) {
    Matrix<Type>* dX_ptr = &dX;
    Matrix<Type>* X_ptr = &X;

    CoolDiff::TensorR2::MatOperators::MatrixScalarMul(alpha, &dX, dX_ptr);
    CoolDiff::TensorR2::MatOperators::MatrixAdd(&X, dX_ptr, X_ptr);
}

auto NetworkPred( Matrix<Type>& W1, Matrix<Type>& W2, Matrix<Type>& W3, Matrix<Type>& W4,
                  Matrix<Type>& b1, Matrix<Type>& b2, Matrix<Type>& b3, Matrix<Type>& b4,
                  Matrix<Type>& O,  Matrix<Type>& X ) {

  auto Layer1 = TanhM(W1*X + b1);
  auto Layer2 = TanhM(W2*Layer1 + b2);
  auto Layer3 = TanhM(W3*Layer2 + b3);
  auto Layer4 = TanhM(W4*Layer3 + b4);
  auto Yhat = TanhM(O*Layer4);

  return Yhat;
}

auto NetworkErr(  Matrix<Type>& W1, Matrix<Type>& W2, Matrix<Type>& W3, Matrix<Type>& W4,
                  Matrix<Type>& b1, Matrix<Type>& b2, Matrix<Type>& b3, Matrix<Type>& b4,
                  Matrix<Type>& O,  Matrix<Type>& X, Type Y ) {
  auto Yhat = NetworkPred(W1, W2, W3, W4, b1, b2, b3, b4, O, X);
  auto error = (Yhat-Y)*100*(Yhat-Y);
  return error;
}

#ifndef USE_COMPLEX_MATH
void NN() {
  constexpr const int N = 256;

  Matrix<Type> X(N, 1);
  X(0,0) = 2;

  Matrix<Type> W1(N, N), W2(2*N, N), W3(N, 2*N), W4(N, N);
  Matrix<Type> b1(N, 1), b2(2*N, 1), b3(N, 1), b4(N, 1);

  Matrix<Type> O(1, N);

  FillRandomWeights(W1); FillRandomWeights(W2); FillRandomWeights(W3); FillRandomWeights(W4);
  FillRandomWeights(b1); FillRandomWeights(b2); FillRandomWeights(b3); FillRandomWeights(b4);
  FillRandomWeights(O);

  Matrix<Expression> Error = NetworkErr(W1, W2, W3, W4, b1, b2, b3, b4, O, X, 10);

  Type alpha = -0.00001;
  for(int i{}; i < 100; ++i) {
    std::cout << "[ERROR]: " << CoolDiff::TensorR2::Eval(Error) << "\n";
    CoolDiff::TensorR2::PreComp(Error);

    auto& dW1 = CoolDiff::TensorR2::DevalR(Error, W1);
    auto& dW2 = CoolDiff::TensorR2::DevalR(Error, W2);
    auto& dW3 = CoolDiff::TensorR2::DevalR(Error, W3);
    auto& dW4 = CoolDiff::TensorR2::DevalR(Error, W4);

    auto& db1 = CoolDiff::TensorR2::DevalR(Error, b1);
    auto& db2 = CoolDiff::TensorR2::DevalR(Error, b2);
    auto& db3 = CoolDiff::TensorR2::DevalR(Error, b3);
    auto& db4 = CoolDiff::TensorR2::DevalR(Error, b4);

    auto& dO = CoolDiff::TensorR2::DevalR(Error, O);

    GDOptimizer(W1, dW1, alpha);
    GDOptimizer(W2, dW2, alpha);
    GDOptimizer(W3, dW3, alpha);
    GDOptimizer(W4, dW4, alpha);

    GDOptimizer(b1, db1, alpha);
    GDOptimizer(b2, db2, alpha);
    GDOptimizer(b3, db3, alpha);
    GDOptimizer(b4, db4, alpha);

    GDOptimizer(O, dO, alpha);
  }

  auto Yhat = NetworkPred(W1, W2, W3, W4, b1, b2, b3, b4, O, X);
  std::cout << CoolDiff::TensorR2::Eval(Yhat) << "\n";

}
#endif


int main(int argc, char **argv) { 
  #ifndef USE_COMPLEX_MATH
    NN();
  #endif
  GNMatrix();
  NonLinearSolve();
  return 0;
}