/**
 * @file test/MatTest.cpp
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
#include <gtest/gtest.h>

// Matrix derivative order test #4
TEST(MatTest, Test15) {
  double epi = 0.001;

  Type DXres1[3][3] = {{(Type)540.0,  (Type)900.0, (Type)1260.0},
                      {(Type)720.0, (Type)1080.0, (Type)1440.0},
                      {(Type)900.0, (Type)1260.0, (Type)1620.0}};

  Matrix<Variable> R1(1,4);
  Matrix<Variable> R2(4,3);
  Matrix<Variable> R3(3,1);
  Matrix<Variable> R4(3,3);

  // Set R1
  for(int i{}; i < 1; i++) {
    for(int j{}; j < 4; j++) {
      R1(i,j) = i+j+1;
    }
  }
  // Set R2
  for(int i{}; i < 4; i++) {
    for(int j{}; j < 3; j++) {
      R2(i,j) = i+2*j+2;
    }
  }
  // Set R3
  for(int i{}; i < 3; i++) {
    for(int j{}; j < 1; j++) {
      R3(i,j) = 1+j+i*2;
    }
  }
  // Set R4
  for(int i{}; i < 3; i++) {
    for(int j{}; j < 3; j++) {
      R4(i,j) = 1;
    }
  }

  // Matrix Expression 
  Matrix<Expression> res = R1*R2;
  res = res*R4*R4;
  res = res*R3;

  // Evaluated at different locations
  const auto& d1 = CoolDiff::Tensor2R::DevalF(res, R1);
  const auto& d4 = CoolDiff::Tensor2R::DevalF(res, R4);
  const auto& d2 = CoolDiff::Tensor2R::DevalF(res, R2);
  const auto& d3 = CoolDiff::Tensor2R::DevalF(res, R3);

  // Verification deval function
  auto verify_deval_function = [&](auto DXres) {
    for (size_t i{}; i < d4.getNumRows(); ++i) {
      for (size_t j{}; j < d4.getNumColumns(); ++j) {
        ASSERT_NEAR(std::abs(d4(i, j)), std::abs(DXres[i][j]), epi);
      }
    }
  };

  verify_deval_function(DXres1);
}

// Matrix derivative order test #3
TEST(MatTest, Test14) {
  double epi = 0.001;

  Type DXres1[3][3] = {{(Type)540.0,  (Type)900.0, (Type)1260.0},
                      {(Type)720.0, (Type)1080.0, (Type)1440.0},
                      {(Type)900.0, (Type)1260.0, (Type)1620.0}};

  Matrix<Variable> R1(1,4);
  Matrix<Variable> R2(4,3);
  Matrix<Variable> R3(3,1);
  Matrix<Variable> R4(3,3);

  // Set R1
  for(int i{}; i < 1; i++) {
    for(int j{}; j < 4; j++) {
      R1(i,j) = i+j+1;
    }
  }
  // Set R2
  for(int i{}; i < 4; i++) {
    for(int j{}; j < 3; j++) {
      R2(i,j) = i+2*j+2;
    }
  }
  // Set R3
  for(int i{}; i < 3; i++) {
    for(int j{}; j < 1; j++) {
      R3(i,j) = 1+j+i*2;
    }
  }
  // Set R4
  for(int i{}; i < 3; i++) {
    for(int j{}; j < 3; j++) {
      R4(i,j) = 1;
    }
  }

  // Matrix Expression 
  Matrix<Expression> res = R1*R2;
  res = res*R4*R4;
  res = res*R3;

  // Evaluated at different locations
  const auto& d1 = CoolDiff::Tensor2R::DevalF(res, R1);
  const auto& d2 = CoolDiff::Tensor2R::DevalF(res, R2);
  const auto& d4 = CoolDiff::Tensor2R::DevalF(res, R4);
  const auto& d3 = CoolDiff::Tensor2R::DevalF(res, R3);

  // Verification deval function
  auto verify_deval_function = [&](auto DXres) {
    for (size_t i{}; i < d4.getNumRows(); ++i) {
      for (size_t j{}; j < d4.getNumColumns(); ++j) {
        ASSERT_NEAR(std::abs(d4(i, j)), std::abs(DXres[i][j]), epi);
      }
    }
  };

  verify_deval_function(DXres1);
}

// Matrix derivative order test #2
TEST(MatTest, Test13) {
  double epi = 0.001;

  Type DXres1[3][3] = {{(Type)540.0,  (Type)900.0, (Type)1260.0},
                      {(Type)720.0, (Type)1080.0, (Type)1440.0},
                      {(Type)900.0, (Type)1260.0, (Type)1620.0}};

  Matrix<Variable> R1(1,4);
  Matrix<Variable> R2(4,3);
  Matrix<Variable> R3(3,1);
  Matrix<Variable> R4(3,3);

  // Set R1
  for(int i{}; i < 1; i++) {
    for(int j{}; j < 4; j++) {
      R1(i,j) = i+j+1;
    }
  }
  // Set R2
  for(int i{}; i < 4; i++) {
    for(int j{}; j < 3; j++) {
      R2(i,j) = i+2*j+2;
    }
  }
  // Set R3
  for(int i{}; i < 3; i++) {
    for(int j{}; j < 1; j++) {
      R3(i,j) = 1+j+i*2;
    }
  }
  // Set R4
  for(int i{}; i < 3; i++) {
    for(int j{}; j < 3; j++) {
      R4(i,j) = 1;
    }
  }

  // Matrix Expression 
  Matrix<Expression> res = R1*R2;
  res = res*R4*R4;
  res = res*R3;

  // Evaluated at different locations
  const auto& d1 = CoolDiff::Tensor2R::DevalF(res, R1);
  const auto& d2 = CoolDiff::Tensor2R::DevalF(res, R2);
  const auto& d3 = CoolDiff::Tensor2R::DevalF(res, R3);
  const auto& d4 = CoolDiff::Tensor2R::DevalF(res, R4);

  // Verification deval function
  auto verify_deval_function = [&](auto DXres) {
    for (size_t i{}; i < d4.getNumRows(); ++i) {
      for (size_t j{}; j < d4.getNumColumns(); ++j) {
        ASSERT_NEAR(std::abs(d4(i, j)), std::abs(DXres[i][j]), epi);
      }
    }
  };

  verify_deval_function(DXres1);
}

// Matrix derivative order test #1
TEST(MatTest, Test12) {
  double epi = 0.001;

  Type DXres1[3][3] = {{(Type)540.0,  (Type)900.0, (Type)1260.0},
                      {(Type)720.0, (Type)1080.0, (Type)1440.0},
                      {(Type)900.0, (Type)1260.0, (Type)1620.0}};

  Matrix<Variable> R1(1,4);
  Matrix<Variable> R2(4,3);
  Matrix<Variable> R3(3,1);
  Matrix<Variable> R4(3,3);

  // Set R1
  for(int i{}; i < 1; i++) {
    for(int j{}; j < 4; j++) {
      R1(i,j) = i+j+1;
    }
  }
  // Set R2
  for(int i{}; i < 4; i++) {
    for(int j{}; j < 3; j++) {
      R2(i,j) = i+2*j+2;
    }
  }
  // Set R3
  for(int i{}; i < 3; i++) {
    for(int j{}; j < 1; j++) {
      R3(i,j) = 1+j+i*2;
    }
  }
  // Set R4
  for(int i{}; i < 3; i++) {
    for(int j{}; j < 3; j++) {
      R4(i,j) = 1;
    }
  }

  // Matrix Expression 
  Matrix<Expression> res = R1*R2;
  res = res*R4*R4;
  res = res*R3;

  // Evaluated at different locations
  const auto& d4 = CoolDiff::Tensor2R::DevalF(res, R4);
  const auto& d1 = CoolDiff::Tensor2R::DevalF(res, R1);
  const auto& d2 = CoolDiff::Tensor2R::DevalF(res, R2);
  const auto& d3 = CoolDiff::Tensor2R::DevalF(res, R3);

  // Verification deval function
  auto verify_deval_function = [&](auto DXres) {
    for (size_t i{}; i < d4.getNumRows(); ++i) {
      for (size_t j{}; j < d4.getNumColumns(); ++j) {
        ASSERT_NEAR(std::abs(d4(i, j)), std::abs(DXres[i][j]), epi);
      }
    }
  };

  verify_deval_function(DXres1);
}

// Matrix softmax
TEST(MatTest, Test11) {
  double epi = 0.001;

  // Eval and Deval result 1
  Type Xres1[2][2] = {{(Type)0.02371293659, (Type)0.4762870634},
                      {(Type)0.00006169728799, (Type)0.4999383027}};

  Type DXres1[2][8] = {{(Type)-0.001124606723, (Type)-0.03444479816, (Type)0.04517665973, (Type)-0.04517665973, (Type)-0.4988753933, (Type)-0.2155552018, (Type)-0.04517665973, (Type)0.04517665973},
                      {(Type)0.0001850614115, (Type)-0.0002776073435, (Type)0.0003084483744, (Type)-0.0003084483744, (Type)-0.5001850614, (Type)-0.2497223927, (Type)-0.0003084483744, (Type)0.0003084483744}};

  // Eval and Deval result 2
  Type Xres2[2][2] = {{(Type)0.001580862439, (Type)0.03175247089},
                      {(Type)0.000004113152533, (Type)0.03332922018}};

  Type DXres2[2][8] = {{(Type)0.00118971617, (Type)-0.001769365731, (Type)0.003011777315, (Type)-0.003011777315, (Type)-0.007856382837, (Type)-0.003786189825, (Type)-0.003011777315, (Type)0.003011777315},
                      {(Type)0.00001562794946, (Type)-0.00001713610539, (Type)0.00002056322496, (Type)-0.00002056322496, (Type)-0.006682294616, (Type)-0.00553841945, (Type)-0.00002056322496, (Type)0.00002056322496}};

  Matrix<Variable> X(1,4); 
  X[0] = 1; X[1] = 2; X[2] = 3; X[3] = 4;

  Matrix<Type> A(2,2);
  A(0,0) = 1; A(0,1) = 2;
  A(1,0) = 4; A(1,1) = 5;

  Matrix<Expression> S(2,2);
  S(0,0) = X[0]; S(0,1) = X[1];
  S(1,0) = X[2]; S(1,1) = X[3];

  Matrix<Expression> E = SoftMax<Axis::COLUMN>(A*S);
  E = E/(X[0]*X[1]);

  // Verification eval function
  auto verify_eval_function = [&](auto Xres) {
    const auto& R = CoolDiff::Tensor2R::Eval(E);
    for (size_t i{}; i < R.getNumRows(); ++i) {
      for (size_t j{}; j < R.getNumColumns(); ++j) {
        ASSERT_NEAR(std::abs(R(i, j)), std::abs(Xres[i][j]), epi);
      }
    }
  };
  
  // Verification deval function
  auto verify_deval_function = [&](auto DXres) {
    const auto& DR = CoolDiff::Tensor2R::DevalF(E, X);
    for (size_t i{}; i < DR.getNumRows(); ++i) {
      for (size_t j{}; j < DR.getNumColumns(); ++j) {
        ASSERT_NEAR(std::abs(DR(i, j)), std::abs(DXres[i][j]), epi);
      }
    }
  };

  verify_eval_function(Xres1);
  verify_deval_function(DXres1);

  X[0] = 5; X[1] = 6; X[2] = 7; X[3] = 8;

  verify_eval_function(Xres2);
  verify_deval_function(DXres2);
}

// Matrix determinant (TODO)
TEST(MatTest, Test10) {
  Matrix<Variable> X(1,4); 
  X[0] = 1; X[1] = 0; X[2] = 0; X[3] = 1;

  Matrix<Type> A(2,2);
  A(0,0) = 1; A(0,1) = 2;
  A(1,0) = 4; A(1,1) = 5;


  Matrix<Expression> S(2,2);
  S(0,0) = X[0]; S(0,1) = X[1];
  S(1,0) = X[2]; S(1,1) = X[3];

  Matrix<Expression> E = det(A*S);
  E = LogM(E);

  CoolDiff::Tensor2R::Eval(E);
  CoolDiff::Tensor2R::DevalF(E,X); 
}

// Matrix inverse and exponential operation (TODO)
TEST(MatTest, Test9) {
  Matrix<Variable> X(1,4); 
  X[0] = 1; X[1] = 2; X[2] = 3; X[3] = 4;

  Matrix<Expression> S(2,2);
  S(0,0) = X[0]; S(0,1) = X[1];
  S(1,0) = X[2]; S(1,1) = X[3];

  Matrix<Type> A(2,2);
  A(0,0) = 1; A(0,1) = 2;
  A(1,0) = 4; A(1,1) = 5;

  Matrix<Expression> E;
  E = (A*inv(MatrixExponential(S))*inv(A));
  E = inv(E)/(S(0,0)*S(1,0));

  CoolDiff::Tensor2R::Eval(E);
  CoolDiff::Tensor2R::DevalF(E,X);
}

// Matrix sin/cos operation
TEST(MatTest, Test8) {
  double epi = 0.001;

  // Eval and Deval result 1
  Type Xres1[2][2] = {{(Type)116.3192408, (Type)181.0049466},
                      {(Type)350.8476951, (Type)527.8106938}};

  Type DXres1[4][4] = {{(Type)20.79150789, (Type)31.33700343, (Type)-21.3781716,
                        (Type)66.48187411},
                      {(Type)34.44941922, (Type)-11.19563462,
                        (Type)10.17733502, (Type)17.47356469},
                      {(Type)86.72069026, (Type)39.62774697,
                        (Type)-10.14312215, (Type)143.395168},
                      {(Type)101.5180748, (Type)-74.948339, (Type)28.19618032,
                        (Type)-4.787774304}};

  // Eval and Deval result 2
  Type Xres2[2][2] = {{(Type)151.2665253, (Type)103.3465801},
                      {(Type)478.2702848, (Type)317.4912399}};

  Type DXres2[4][4] = {{(Type)-22.3638443, (Type)7.66467508, (Type)-32.47476611,
                        (Type)30.17775842},
                      {(Type)60.24722736, (Type)-6.085873292,
                        (Type)23.77664818, (Type)21.60863041},
                      {(Type)56.61282765, (Type)16.96222503,
                        (Type)-18.65939104, (Type)89.13394705},
                      {(Type)176.5870944, (Type)-46.04305063,
                        (Type)74.30870851, (Type)48.47765126}};

  Variable x1{1}, x2{2}, x3{3}, x4{4};
  Matrix<Variable> X(2, 2);
  X(0, 0) = x1;
  X(0, 1) = x2;
  X(1, 0) = x3;
  X(1, 1) = x4;

  Matrix<Type> A(2, 2);
  A(0, 0) = -1;
  A(0, 1) = 3;
  A(1, 0) = 5;
  A(1, 1) = 6;

  MatExpression E = A * CosM(SinM(X)) + X;
  E = A * E * X + CosM(2 * X) * E;

  // Verification eval function
  auto verify_eval_function = [&](auto Xres) {
    const auto& R = CoolDiff::Tensor2R::Eval(E);
    for (size_t i{}; i < R.getNumRows(); ++i) {
      for (size_t j{}; j < R.getNumColumns(); ++j) {
        ASSERT_NEAR(std::abs(R(i, j)), std::abs(Xres[i][j]), epi);
      }
    }
  };

  // Verification deval function
  auto verify_deval_function = [&](auto DXres) {
    const auto& DR = CoolDiff::Tensor2R::DevalF(E, X);
    for (size_t i{}; i < DR.getNumRows(); ++i) {
      for (size_t j{}; j < DR.getNumColumns(); ++j) {
        ASSERT_NEAR(std::abs(DR(i, j)), std::abs(DXres[i][j]), epi);
      }
    }
  };

  verify_eval_function(Xres1);
  verify_deval_function(DXres1);

  x1 = 4;
  x2 = 3;
  x3 = 2;
  x4 = 1;

  verify_eval_function(Xres2);
  verify_deval_function(DXres2);
}

// Matrix convolution operation (with variable change)
TEST(MatTest, Test7) {
  // Result set 1
  Type Eres1[1][1] = {(Type)-182};
  Type DW1res1[2][2] = {{(Type)-26, (Type)-26}, {(Type)-26, (Type)-26}};
  Type DW2res1[2][2] = {{(Type)-91, (Type)-91}, {(Type)-91, (Type)-91}};

  // Result set 2
  Type Eres2[1][1] = {(Type)1040};
  Type DW1res2[2][2] = {{(Type)104, (Type)104}, {(Type)104, (Type)104}};
  Type DW2res2[2][2] = {{(Type)-130, (Type)-130}, {(Type)-130, (Type)-130}};

  Matrix<Type> X(3, 3);
  Matrix<Variable> W1(2, 2), W2(2, 2);

  X(0, 0) = -1;
  X(0, 1) = 2;
  X(0, 2) = -3;
  X(1, 0) = 4;
  X(1, 1) = -13;
  X(1, 2) = 5;
  X(2, 0) = -6;
  X(2, 1) = 7;
  X(2, 2) = -8;

  W1(0, 0) = -1;
  W1(0, 1) = 2;
  W1(1, 0) = -3;
  W1(1, 1) = 9;

  W2(0, 0) = -3;
  W2(0, 1) = 4;
  W2(1, 0) = -5;
  W2(1, 1) = 6;

  /* TODO - Fix this issue
  Matrix<Expression> E;
  E = conv(X, W2, 1, 1, 1, 1);
  E = conv(E, W1, 1, 1, 1, 1);
  E = Sigma(E);
  */

  Matrix<Expression> Exp1, Exp2;
  Exp1 = conv(X, W2, 1, 1, 1, 1);
  Exp2 = conv(Exp1, W1, 1, 1, 1, 1);
  Matrix<Expression> E = Sigma(Exp2);

  // Verification eval function
  auto verify_eval_function = [&](auto Xres) {
    const auto& R = CoolDiff::Tensor2R::Eval(E);
    for (size_t i{}; i < R.getNumRows(); ++i) {
      for (size_t j{}; j < R.getNumColumns(); ++j) {
        if (R(i, j) != Xres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Verification deval function
  auto verify_deval_function = [&](auto DXres, auto &X) {
    const auto& DR = CoolDiff::Tensor2R::DevalF(E, X);
    for (size_t i{}; i < DR.getNumRows(); ++i) {
      for (size_t j{}; j < DR.getNumColumns(); ++j) {
        if (DR(i, j) != DXres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Assert test
  ASSERT_TRUE(verify_eval_function(Eres1));
  ASSERT_TRUE(verify_deval_function(DW1res1, W1));
  ASSERT_TRUE(verify_deval_function(DW2res1, W2));

  W1(0, 0) = 1;
  W1(0, 1) = 4;
  W1(1, 0) = -1;
  W1(1, 1) = 6;

  W2(0, 0) = -1;
  W2(0, 1) = -8;
  W2(1, 0) = 3;
  W2(1, 1) = -2;

  // Assert test
  ASSERT_TRUE(verify_eval_function(Eres2));
  ASSERT_TRUE(verify_deval_function(DW1res2, W1));
  ASSERT_TRUE(verify_deval_function(DW2res2, W2));
}

// Matrix trace operation (with variable change)
TEST(MatTest, Test6) {
  // Eval and Deval result 1
  Type Xres1[1][1] = {(Type)2244.0};

  Type DXres1[3][2] = {
      {(Type)830.0, (Type)3.0}, {(Type)851.0, (Type)0}, {(Type)738.0, (Type)0}};

  // Eval and Deval result 2
  Type Xres2[1][1] = {(Type)1980.0};

  Type DXres2[3][2] = {
      {(Type)851.0, (Type)3.0}, {(Type)573.0, (Type)0}, {(Type)672.0, (Type)0}};

  Variable x1{1}, x2{2}, x3{3};
  Matrix<Variable> X(3, 1);
  X(0, 0) = x1;
  X(1, 0) = x2;
  X(2, 0) = x3;

  Matrix<Type> A(2, 3);
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(0, 2) = 3;
  A(1, 0) = 7;
  A(1, 1) = 5;
  A(1, 2) = 9;

  Matrix<Expression> Y(2, 2);
  Y(0, 0) = x1 + x2;
  Y(0, 1) = x1 * x2 - x3;
  Y(1, 0) = x2 - x1 + x3;
  Y(1, 1) = x2 + x1 + x3;

  Matrix<Type> Z(1, 2);
  Z(0, 0) = 3;
  Z(0, 1) = 7;

  Matrix<Expression> E = (Y * A * X) * Z + 2 * x1 * x2;
  E = E - x1 * x2 + x3;
  E = trace(E);

  // Verification eval function
  auto verify_eval_function = [&](auto Xres) {
    const auto& R = CoolDiff::Tensor2R::Eval(E);
    for (size_t i{}; i < R.getNumRows(); ++i) {
      for (size_t j{}; j < R.getNumColumns(); ++j) {
        if (R(i, j) != Xres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Verification deval function
  auto verify_deval_function = [&](auto DXres) {
    const auto& DR = CoolDiff::Tensor2R::DevalF(E, X);
    for (size_t i{}; i < DR.getNumRows(); ++i) {
      for (size_t j{}; j < DR.getNumColumns(); ++j) {
        if (DR(i, j) != DXres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Assert test
  ASSERT_TRUE(verify_eval_function(Xres1));
  ASSERT_TRUE(verify_deval_function(DXres1));

  // Change of variable values
  x1 = -1;
  x2 = 3;
  x3 = 4;

  // Assert test
  ASSERT_TRUE(verify_eval_function(Xres2));
  ASSERT_TRUE(verify_deval_function(DXres2));
}

// Matrix sigma operation (with variable change)
TEST(MatTest, Test5) {
  // Eval and Deval result 1
  Type Xres1[1][1] = {(Type)318.0};

  Type DXres1[3][2] = {
      {(Type)174.0, (Type)3.0}, {(Type)155.0, (Type)0}, {(Type)80.0, (Type)0}};

  // Eval and Deval result 2
  Type Xres2[1][1] = {(Type)126.0};

  Type DXres2[3][2] = {
      {(Type)179.0, (Type)3.0}, {(Type)49.0, (Type)0}, {(Type)38.0, (Type)0}};

  Variable x1{1}, x2{2}, x3{3};
  Matrix<Variable> X(3, 1);
  X(0, 0) = x1;
  X(1, 0) = x2;
  X(2, 0) = x3;

  Matrix<Type> A(2, 3);
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(0, 2) = 3;
  A(1, 0) = 7;
  A(1, 1) = 5;
  A(1, 2) = 9;

  Matrix<Expression> Y(2, 2);
  Y(0, 0) = x1 + x2;
  Y(0, 1) = x1 * x2 - x3;
  Y(1, 0) = x2 - x1 + x3;
  Y(1, 1) = x2 + x1 + x3;

  Matrix<Expression> E = Y * A * X;
  E = Sigma(E);

  // Verification eval function
  auto verify_eval_function = [&](auto Xres) {
    const auto& R = CoolDiff::Tensor2R::Eval(E);
    for (size_t i{}; i < R.getNumRows(); ++i) {
      for (size_t j{}; j < R.getNumColumns(); ++j) {
        if (R(i, j) != Xres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Verification deval function
  auto verify_deval_function = [&](auto DXres) {
    const auto& DR = CoolDiff::Tensor2R::DevalF(E, X);
    for (size_t i{}; i < DR.getNumRows(); ++i) {
      for (size_t j{}; j < DR.getNumColumns(); ++j) {
        if (DR(i, j) != DXres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Assert test
  ASSERT_TRUE(verify_eval_function(Xres1));
  ASSERT_TRUE(verify_deval_function(DXres1));

  // Change of variable values
  x1 = -1;
  x2 = 3;
  x3 = 4;

  // Assert test
  ASSERT_TRUE(verify_eval_function(Xres2));
  ASSERT_TRUE(verify_deval_function(DXres2));
}

// Matrix with scalar expressions (With Hadamard operator and parameter change)
TEST(MatTest, Test4) {
  // Eval and Deval result - 1
  Type Xres1[2][2] = {{(Type)-368.0, (Type)-640.0},
                      {(Type)-720.0, (Type)-512.0}};

  Type DXres1[4][4] = {
      {(Type)-352.0, (Type)-376.0, (Type)-576.0, (Type)-672.0},
      {(Type)-160.0, (Type)0, (Type)-320.0, (Type)0},
      {(Type)-720.0, (Type)-1080.0, (Type)-512.0, (Type)-1216.0},
      {(Type)-432.0, (Type)144.0, (Type)-512.0, (Type)256.0}};

  // Eval and Deval result - 2
  Type Xres2[2][2] = {{(Type)-468.0, (Type)-720.0},
                      {(Type)-540.0, (Type)288.0}};

  Type DXres2[4][4] = {
      {(Type)-432.0, (Type)-486.0, (Type)-576.0, (Type)-792.0},
      {(Type)-240.0, (Type)0, (Type)-480.0, (Type)0},
      {(Type)-540.0, (Type)-1350.0, (Type)288.0, (Type)-1296.0},
      {(Type)-612.0, (Type)324.0, (Type)-672.0, (Type)576.0}};

  Variable x1{1}, x2{2}, x3{3}, x4{4};
  Parameter s{2};

  // Variable X
  Matrix<Variable> X(2, 2);
  X(0, 0) = x1;
  X(0, 1) = x2;
  X(1, 0) = x3;
  X(1, 1) = x4;

  // Get matrix via factory
  Matrix<Type> A = Matrix<Type>::MatrixFactory::CreateMatrix(2, 2);
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;

  Matrix<Expression> Y(2, 2);
  Y(0, 0) = x1 + x2;
  Y(0, 1) = x1 * x2 - x3;
  Y(1, 0) = x2 - x1 + x3;
  Y(1, 1) = x2 / x4;

  Matrix<Expression> E = s * X * A * x1 - x1 * 5 * x2 * 2 * x3;
  E = E * x2 * 2 * s;
  E = E ^ A;

  // Verification eval function
  auto verify_eval_function = [&](auto Xres) {
    const auto& R = CoolDiff::Tensor2R::Eval(E);
    for (size_t i{}; i < R.getNumRows(); ++i) {
      for (size_t j{}; j < R.getNumColumns(); ++j) {
        if (R(i, j) != Xres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Verification deval function
  auto verify_deval_function = [&](auto DXres) {
    const auto& DR = CoolDiff::Tensor2R::DevalF(E, X);
    for (size_t i{}; i < DR.getNumRows(); ++i) {
      for (size_t j{}; j < DR.getNumColumns(); ++j) {
        if (DR(i, j) != DXres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Assert test
  ASSERT_TRUE(verify_eval_function(Xres1));
  ASSERT_TRUE(verify_deval_function(DXres1));

  // Parameter change
  s = 3;

  // Assert test
  ASSERT_TRUE(verify_eval_function(Xres2));
  ASSERT_TRUE(verify_deval_function(DXres2));
}

// Matrix with scalar expressions (With parameter change)
TEST(MatTest, Test3) {

  // Eval and Deval result - 1
  Type Xres1[2][2] = {{(Type)-368.0, (Type)-320.0},
                      {(Type)-240.0, (Type)-128.0}};

  Type DXres1[4][4] = {{(Type)-352.0, (Type)-376.0, (Type)-288.0, (Type)-336.0},
                      {(Type)-160.0, (Type)0, (Type)-160.0, (Type)0},
                      {(Type)-240.0, (Type)-360.0, (Type)-128.0, (Type)-304.0},
                      {(Type)-144.0, (Type)48.0, (Type)-128.0, (Type)64.0}};

  // Eval and Deval result - 2
  Type Xres2[2][2] = {{(Type)-468.0, (Type)-360.0}, {(Type)-180.0, (Type)72.0}};

  Type DXres2[4][4] = {{(Type)-432.0, (Type)-486.0, (Type)-288.0, (Type)-396.0},
                      {(Type)-240.0, (Type)0, (Type)-240.0, (Type)0},
                      {(Type)-180.0, (Type)-450.0, (Type)72.0, (Type)-324.0},
                      {(Type)-204.0, (Type)108.0, (Type)-168.0, (Type)144.0}};

  Variable x1{1}, x2{2}, x3{3}, x4{4};
  Parameter s{2};

  // Variable X
  Matrix<Variable> X(2, 2);
  X(0, 0) = x1;
  X(0, 1) = x2;
  X(1, 0) = x3;
  X(1, 1) = x4;

  // Get matrix via factory
  Matrix<Type> A = Matrix<Type>::MatrixFactory::CreateMatrix(2, 2);
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;

  Matrix<Expression> Y(2, 2);
  Y(0, 0) = x1 + x2;
  Y(0, 1) = x1 * x2 - x3;
  Y(1, 0) = x2 - x1 + x3;
  Y(1, 1) = x2 / x4;

  Matrix<Expression> E = s * X * A * x1 - x1 * 5 * x2 * 2 * x3;
  E = E * x2 * 2 * s;

  // Verification eval function
  auto verify_eval_function = [&](auto Xres) {
    const auto& R = CoolDiff::Tensor2R::Eval(E);
    for (size_t i{}; i < R.getNumRows(); ++i) {
      for (size_t j{}; j < R.getNumColumns(); ++j) {
        if (R(i, j) != Xres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Verification deval function
  auto verify_deval_function = [&](auto DXres) {
    const auto& DR = CoolDiff::Tensor2R::DevalF(E, X);
    for (size_t i{}; i < DR.getNumRows(); ++i) {
      for (size_t j{}; j < DR.getNumColumns(); ++j) {
        if (DR(i, j) != DXres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Assert test
  ASSERT_TRUE(verify_eval_function(Xres1));
  ASSERT_TRUE(verify_deval_function(DXres1));

  // Parameter change
  s = 3;

  // Assert test
  ASSERT_TRUE(verify_eval_function(Xres2));
  ASSERT_TRUE(verify_deval_function(DXres2));
}

// Complicated matrix multiplication example (Change of values)
TEST(MatTest, Test2) {

  // Eval and Deval result - 1
  Type Xres1[2][1] = {{(Type)59020.0}, {(Type)136776.0}};

  Type DXres1[4][2] = {{(Type)25636.0, (Type)29072.0},
                      {(Type)14748.0, (Type)26832.0},
                      {(Type)16920.0, (Type)21848.0},
                      {(Type)52960.0, (Type)81504.0}};

  // Eval and Deval result - 2
  Type Xres2[2][1] = {{(Type)2686124.0}, {(Type)3667720.0}};

  Type DXres2[4][2] = {{(Type)417764.0, (Type)414744.0},
                      {(Type)308172.0, (Type)500504.0},
                      {(Type)218696.0, (Type)250800.0},
                      {(Type)679264.0, (Type)913704.0}};

  // Variable X
  Matrix<Variable> X(2, 2);
  X(0, 0) = 1;
  X(0, 1) = 2;
  X(1, 0) = 3;
  X(1, 1) = 4;

  // Get matrix via factory
  Matrix<Type> Y = Matrix<Type>::MatrixFactory::CreateMatrix(2, 2);
  Y(0, 0) = 1;
  Y(0, 1) = 2;
  Y(1, 0) = 3;
  Y(1, 1) = 4;

  // Matrix expression
  Matrix<Expression> Z(2, 2);
  Z(0, 0) = X(0, 0) + X(1, 0);
  Z(0, 1) = X(1, 1) + X(0, 1);
  Z(1, 0) = X(1, 0) + X(1, 1);
  Z(1, 1) = X(1, 1);

  // Column type matrix
  Matrix<Type> M(2, 1);
  M(0, 0) = 1;
  M(1, 0) = 5;

  // Matrix expression
  Matrix<Expression> E = Z * Z * Z + X;
  E = E * Y + X;
  E = X * E * M;

  // Verification eval function
  auto verify_eval_function = [&](auto Xres) {
    const auto& R = CoolDiff::Tensor2R::Eval(E);
    for (size_t i{}; i < R.getNumRows(); ++i) {
      for (size_t j{}; j < R.getNumColumns(); ++j) {
        if (R(i, j) != Xres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Verification deval function
  auto verify_deval_function = [&](auto DXres) {
    const auto& DR = CoolDiff::Tensor2R::DevalF(E, X);
    for (size_t i{}; i < DR.getNumRows(); ++i) {
      for (size_t j{}; j < DR.getNumColumns(); ++j) {
        if (DR(i, j) != DXres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Assert test
  ASSERT_TRUE(verify_eval_function(Xres1));
  ASSERT_TRUE(verify_deval_function(DXres1));

  // Change of values
  X(0, 0) = 5;
  X(0, 1) = 6;
  X(1, 0) = 7;
  X(1, 1) = 8;

  // Assert test
  ASSERT_TRUE(verify_eval_function(Xres2));
  ASSERT_TRUE(verify_deval_function(DXres2));
}

// Matrix multiplication and addition (Repeatability test)
TEST(MatTest, Test1) {
  // Eval and Deval result
  Type Xres[2][2] = {{(Type)-35.0, (Type)-270.0}, {(Type)96.0, (Type)630.0}};

  Type DXres[4][4] = {{(Type)26.0, (Type)0, (Type)25.0, (Type)22.0},
                      {(Type)-19.0, (Type)-2.0, (Type)0, (Type)-34.0},
                      {(Type)9.0, (Type)3.0, (Type)20.0, (Type)26.0},
                      {(Type)35.0, (Type)3.0, (Type)30.0, (Type)50.0}};

  // Variable X
  Matrix<Variable> X(2, 2);
  X(0, 0) = 1;
  X(0, 1) = 5;
  X(1, 0) = 3;
  X(1, 1) = 20;

  // Identity matrix
  Matrix<Type> A = Matrix<Type>::MatrixFactory::CreateMatrix(2, 2);
  A(0, 0) = 1;
  A(0, 1) = 0;
  A(1, 0) = 0;
  A(1, 1) = 1;

  // Y matrix
  Matrix<Expression> Y(2, 2);
  Y(0, 0) = X(0, 0) + X(1, 1);
  Y(0, 1) = X(0, 0) - X(1, 1);
  Y(1, 0) = X(1, 0) + X(1, 0);
  Y(1, 1) = X(0, 0) + X(1, 1) + X(0, 1) + X(1, 0);

  // Matrix Multiplication/Addition expression
  Matrix<Expression> E = A * Y;
  E = E * X + X;

  // Verification eval function
  auto verify_eval_function = [&](auto Xres) {
    const auto& R = CoolDiff::Tensor2R::Eval(E);
    for (size_t i{}; i < R.getNumRows(); ++i) {
      for (size_t j{}; j < R.getNumColumns(); ++j) {
        if (R(i, j) != Xres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Verification deval function
  auto verify_deval_function = [&](auto DXres) {
    const auto& DR = CoolDiff::Tensor2R::DevalF(E, X);
    for (size_t i{}; i < DR.getNumRows(); ++i) {
      for (size_t j{}; j < DR.getNumColumns(); ++j) {
        if (DR(i, j) != DXres[i][j]) {
          return false;
        }
      }
    }
    return true;
  };

  // Repeatability test
  ASSERT_TRUE(verify_eval_function(Xres));
  ASSERT_TRUE(verify_deval_function(DXres));
  ASSERT_TRUE(verify_deval_function(DXres));
  ASSERT_TRUE(verify_eval_function(Xres));
}

int main(int argc, char **argv) {
#if defined(USE_COMPLEX_MATH)
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#else
  return RUN_ALL_TESTS();
#endif
}
