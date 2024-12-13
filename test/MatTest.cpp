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

#include <gtest/gtest.h>
#include "CoolDiff.hpp"

// Matrix transpose operation
TEST(MatTest, Test5) {
    // Eval and Deval result
    Type Xres[1][1] = {(Type)102408.0};

    Type DXres[3][2] = {{(Type)48252.0,   (Type)3.0},
                        {(Type)61206.0,   (Type)0},
                        {(Type)79536.0,   (Type)0}};

    Variable x1{1}, x2{2}, x3{3};
    Matrix<Variable> X(3,1);
    X(0,0) = x1; X(1,0) = x2; X(2,0) = x3; 

    Matrix<Type> A(2,3);
    A(0,0) = 1; A(0,1) = 2; A(0,2) = 3;
    A(1,0) = 7; A(1,1) = 5; A(1,2) = 9;

    Matrix<Expression> Y(2,2);
    Y(0,0) = x1+x2; Y(0,1) = x1*x2-x3;
    Y(1,0) = x2-x1+x3; Y(1,1) = x2+x1+x3;

    Matrix<Expression> E = transpose(Y*A*X)*(Y*A*X);
    E = E + 2*x1*x2;

    // Verification eval function 
    auto verify_eval_function = [&](auto Xres) {
        auto& R = Eval(E);
        for(size_t i{}; i < R.getNumRows(); ++i){
            for(size_t j{}; j < R.getNumColumns(); ++j) {
                if(R(i,j) != Xres[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }; 

    // Verification deval function
    auto verify_deval_function = [&](auto DXres) {
        auto& DR = DevalF(E, X);
        for(size_t i{}; i < DR.getNumRows(); ++i){
            for(size_t j{}; j < DR.getNumColumns(); ++j) {
                if(DR(i,j) != DXres[i][j]) {
                    return false;
                }
            }
        }
        return true;
    };

    // Assert test
    ASSERT_TRUE(verify_eval_function(Xres)); 
    ASSERT_TRUE(verify_deval_function(DXres));

}

// Matrix with scalar expressions (With Hadamard operator and parameter change)
TEST(MatTest, Test4) {
  // Eval and Deval result - 1
 Type Xres1[2][2] = {{(Type)-368.0, (Type)-640.0},
                     {(Type)-720.0, (Type)-512.0}};

 Type DXres1[4][4] = {{ (Type)-352.0,  (Type)-376.0, (Type)-576.0,  (Type)-672.0},
                      { (Type)-160.0,       (Type)0, (Type)-320.0,       (Type)0},
                      { (Type)-720.0, (Type)-1080.0, (Type)-512.0, (Type)-1216.0},
                      { (Type)-432.0,   (Type)144.0, (Type)-512.0,   (Type)256.0}};

 // Eval and Deval result - 2
 Type Xres2[2][2] = {{(Type)-468.0, (Type)-720.0},
                     {(Type)-540.0,  (Type)288.0}};

 Type DXres2[4][4] = {{(Type)-432.0,  (Type)-486.0, (Type)-576.0,  (Type)-792.0},
                      {(Type)-240.0,       (Type)0, (Type)-480.0,       (Type)0},
                      {(Type)-540.0, (Type)-1350.0,  (Type)288.0, (Type)-1296.0},
                      {(Type)-612.0,   (Type)324.0, (Type)-672.0,   (Type)576.0}};

  Variable x1{1}, x2{2}, x3{3}, x4{4};
  Parameter s{2};

  // Variable X
  Matrix<Variable> X(2, 2);
  X(0, 0) = x1; X(0, 1) = x2;
  X(1, 0) = x3; X(1, 1) = x4;

  // Get matrix via factory
  Matrix<Type> A = Matrix<Type>::MatrixFactory::CreateMatrix(2, 2);
  A(0, 0) = 1; A(0, 1) = 2;
  A(1, 0) = 3; A(1, 1) = 4;

  Matrix<Expression> Y(2,2);
  Y(0,0) = x1+x2; Y(0,1) = x1*x2-x3;
  Y(1,0) = x2-x1+x3; Y(1,1) = x2/x4;

  Matrix<Expression> E = s*X*A*x1 - x1*5*x2*2*x3;
  E = E*x2*2*s;
  E = E^A;

   // Verification eval function 
 auto verify_eval_function = [&](auto Xres) {
    auto& R = Eval(E);
    for(size_t i{}; i < R.getNumRows(); ++i){
        for(size_t j{}; j < R.getNumColumns(); ++j) {
            if(R(i,j) != Xres[i][j]) {
                return false;
            }
        }
    }
    return true;
 }; 

 // Verification deval function
 auto verify_deval_function = [&](auto DXres) {
    auto& DR = DevalF(E, X);
    for(size_t i{}; i < DR.getNumRows(); ++i){
        for(size_t j{}; j < DR.getNumColumns(); ++j) {
            if(DR(i,j) != DXres[i][j]) {
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
                      {(Type)-160.0,      (Type)0, (Type)-160.0,      (Type)0},
                      {(Type)-240.0, (Type)-360.0, (Type)-128.0, (Type)-304.0},
                      {(Type)-144.0,   (Type)48.0, (Type)-128.0,   (Type)64.0}};

 // Eval and Deval result - 2
 Type Xres2[2][2] = {{(Type)-468.0, (Type)-360.0},
                     {(Type)-180.0, (Type)72.0}};

 Type DXres2[4][4] = {{(Type)-432.0, (Type)-486.0, (Type)-288.0, (Type)-396.0},
                      {(Type)-240.0,      (Type)0, (Type)-240.0,      (Type)0},
                      {(Type)-180.0, (Type)-450.0, (Type)72.0, (Type)-324.0},
                      {(Type)-204.0,   (Type)108.0, (Type)-168.0,   (Type)144.0}};

  Variable x1{1}, x2{2}, x3{3}, x4{4};
  Parameter s{2};

  // Variable X
  Matrix<Variable> X(2, 2);
  X(0, 0) = x1; X(0, 1) = x2;
  X(1, 0) = x3; X(1, 1) = x4;

  // Get matrix via factory
  Matrix<Type> A = Matrix<Type>::MatrixFactory::CreateMatrix(2, 2);
  A(0, 0) = 1; A(0, 1) = 2;
  A(1, 0) = 3; A(1, 1) = 4;

  Matrix<Expression> Y(2,2);
  Y(0,0) = x1+x2; Y(0,1) = x1*x2-x3;
  Y(1,0) = x2-x1+x3; Y(1,1) = x2/x4;

  Matrix<Expression> E = s*X*A*x1 - x1*5*x2*2*x3;
  E = E*x2*2*s;

   // Verification eval function 
 auto verify_eval_function = [&](auto Xres) {
    auto& R = Eval(E);
    for(size_t i{}; i < R.getNumRows(); ++i){
        for(size_t j{}; j < R.getNumColumns(); ++j) {
            if(R(i,j) != Xres[i][j]) {
                return false;
            }
        }
    }
    return true;
 }; 

 // Verification deval function
 auto verify_deval_function = [&](auto DXres) {
    auto& DR = DevalF(E, X);
    for(size_t i{}; i < DR.getNumRows(); ++i){
        for(size_t j{}; j < DR.getNumColumns(); ++j) {
            if(DR(i,j) != DXres[i][j]) {
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
 Type Xres1[2][1] = { {(Type)59020.0}, {(Type)136776.0} };

 Type DXres1[4][2] = { { (Type)25636.0, (Type)29072.0 },
                       { (Type)14748.0, (Type)26832.0 },
                       { (Type)16920.0, (Type)21848.0 },
                       { (Type)52960.0, (Type)81504.0 } };

 // Eval and Deval result - 2
 Type Xres2[2][1] = { {(Type)2686124.0}, {(Type)3667720.0} };

 Type DXres2[4][2] = { {(Type)417764.0, (Type)414744.0 },
                       { (Type)308172.0, (Type)500504.0 },
                       { (Type)218696.0, (Type)250800.0 },
                       { (Type)679264.0, (Type)913704.0 } };

  // Variable X
  Matrix<Variable> X(2, 2);
  X(0, 0) = 1; X(0, 1) = 2;
  X(1, 0) = 3; X(1, 1) = 4;

  // Get matrix via factory
  Matrix<Type> Y = Matrix<Type>::MatrixFactory::CreateMatrix(2, 2);
  Y(0, 0) = 1; Y(0, 1) = 2;
  Y(1, 0) = 3; Y(1, 1) = 4;

  // Matrix expression
  Matrix<Expression> Z(2, 2);
  Z(0, 0) = X(0, 0) + X(1, 0); Z(0, 1) = X(1, 1) + X(0, 1);
  Z(1, 0) = X(1, 0) + X(1, 1); Z(1, 1) = X(1, 1);

  // Column type matrix 
  Matrix<Type> M(2, 1);
  M(0, 0) = 1; M(1, 0) = 5;

  // Matrix expression
  Matrix<Expression> E = Z * Z * Z + X;
  E = E * Y + X;
  E = X * E * M;

 // Verification eval function 
 auto verify_eval_function = [&](auto Xres) {
    const auto& R = Eval(E);
    for(size_t i{}; i < R.getNumRows(); ++i){
        for(size_t j{}; j < R.getNumColumns(); ++j) {
            if(R(i,j) != Xres[i][j]) {
                return false;
            }
        }
    }
    return true;
 }; 

 // Verification deval function
 auto verify_deval_function = [&](auto DXres) {
    const auto& DR = DevalF(E, X);
    for(size_t i{}; i < DR.getNumRows(); ++i){
        for(size_t j{}; j < DR.getNumColumns(); ++j) {
            if(DR(i,j) != DXres[i][j]) {
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
 X(0, 0) = 5; X(0, 1) = 6;
 X(1, 0) = 7; X(1, 1) = 8;

 // Assert test
 ASSERT_TRUE(verify_eval_function(Xres2)); 
 ASSERT_TRUE(verify_deval_function(DXres2));

}

// Matrix multiplication and addition (Repeatability test)
TEST(MatTest, Test1) {
 // Eval and Deval result 
 Type Xres[2][2] = {{(Type)-35.0, (Type)-270.0},
                    {(Type)96.0,  (Type)630.0}};

 Type DXres[4][4] = {{(Type)26.0, (Type)0, (Type)25.0,  (Type)22.0},
                     {(Type)-19.0, (Type)-2.0, (Type)0, (Type)-34.0},
                     {(Type)9.0, (Type)3.0, (Type)20.0, (Type)26.0},
                     {(Type)35.0,  (Type)3.0, (Type)30.0,  (Type)50.0}};


 // Variable X
 Matrix<Variable> X(2, 2);
 X(0, 0) = 1; X(0, 1) = 5;
 X(1, 0) = 3; X(1, 1) = 20;

 // Identity matrix
 Matrix<Type> A = Matrix<Type>::MatrixFactory::CreateMatrix(2, 2);
 A(0, 0) = 1; A(0, 1) = 0;
 A(1, 0) = 0; A(1, 1) = 1;  

 // Y matrix
 Matrix<Expression> Y(2, 2);
 Y(0, 0) = X(0, 0) + X(1, 1); Y(0, 1) = X(0, 0) - X(1, 1);
 Y(1, 0) = X(1, 0) + X(1, 0); Y(1, 1) = X(0, 0) + X(1, 1) + X(0, 1) + X(1, 0);
 
 // Matrix Multiplication/Addition expression 
 Matrix<Expression> E = A * Y;
 E = E * X + X;

 // Verification eval function 
 auto verify_eval_function = [&](auto Xres) {
    const auto& R = Eval(E);
    for(size_t i{}; i < R.getNumRows(); ++i){
        for(size_t j{}; j < R.getNumColumns(); ++j) {
            if(R(i,j) != Xres[i][j]) {
                return false;
            }
        }
    }
    return true;
 }; 

 // Verification deval function
 auto verify_deval_function = [&](auto DXres) {
    const auto& DR = DevalF(E, X);
    for(size_t i{}; i < DR.getNumRows(); ++i){
        for(size_t j{}; j < DR.getNumColumns(); ++j) {
            if(DR(i,j) != DXres[i][j]) {
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
