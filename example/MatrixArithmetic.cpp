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

int main(int argc, char **argv)
{
    Variable x1{2}, x2{3};
    Matrix<Parameter> m = CreateMatrix<Parameter>(3, 2);
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(1, 0) = 3;
    m(1, 1) = 4;

    Matrix<Expression> m1{3, 2};
    m1(0, 0) = x1 + x2;
    m1(0, 1) = x1 * x2;
    m1(1, 0) = x1 / x2;
    m1(1, 1) = x1 - x2;

    Matrix<Expression> sum = m1 + m;
    sum = sum + m;

    auto *res = sum.eval();

    Matrix<Type> ms(19,19);
    ms(2,3) = 1;
    std::cout << IsZeroMatrix(ms) << " " << *res << "\n";
}