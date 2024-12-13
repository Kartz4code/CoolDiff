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
#include <eigen3/Eigen/Dense>
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

// 2D data matching
void GNMatrix() {
  // Load data
  auto data = LoadData();

  // Matrix variables
  Matrix<Variable> V(1, 3);
  V[0] = 0.5142;
  V[1] = 20;
  V[2] = -10;

  // Rotation and translation matrices
  Matrix<Expression> R(2, 2), t(2, 1);
  R(0, 0) = cos(V[0]);
  R(0, 1) = sin(V[0]);
  R(1, 0) = -sin(V[0]);
  R(1, 1) = cos(V[0]);
  t[0] = V[1];
  t[1] = V[2];

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

  std::cout << "Computed values: " << Eval(V);
  std::cout << "Actual values: " << (Type)3.14159 / 2 << " " << (Type)5 << " "
            << (Type)-2 << "\n";
}

int main(int argc, char **argv) {
  GNMatrix();
  return 0;
}