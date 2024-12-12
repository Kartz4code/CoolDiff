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
 auto verify_eval_function = [&]() {
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
 auto verify_deval_function = [&]() {
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
 ASSERT_TRUE(verify_eval_function()); 
 ASSERT_TRUE(verify_deval_function());       
 ASSERT_TRUE(verify_deval_function());   
 ASSERT_TRUE(verify_eval_function()); 
 
}

int main(int argc, char **argv) {
  #if defined(USE_COMPLEX_MATH)
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  #else
    return RUN_ALL_TESTS();
  #endif
}
