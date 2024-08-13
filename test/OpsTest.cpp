/**
 * @file test/OpsTest.cpp
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
#include <iostream>
#include <vector>

#include "CommonFunctions.hpp"

// Evaluation test
TEST(OpsTest, BasicBinaryOps1)
{
    Variable x1{1}, x2{2}, x3{-3}, x4{24}, x5{-99};
    Expression y = 9 + (x1 + x2) * (x3 + x4 * x5) * 3 - 2 * x1 * 4 - 10;

    Type result{-21420};

    ASSERT_EQ(result, Eval(y));
}

// Forward AD test
TEST(OpsTest, BasicBinaryOps2)
{
    Variable x1{1}, x2{2}, x3{-3}, x4{24}, x5{-99};
    Expression y = 9 + (x1 + x2) * (x3 + x4 * x5) * 3 - 2 * x1 * 4 - 10;

    Type results[5] = {-7145, -7137, 9, -891, 216};

    ASSERT_EQ(results[0], DevalF(y, x1));
    ASSERT_EQ(results[1], DevalF(y, x2));
    ASSERT_EQ(results[2], DevalF(y, x3));
    ASSERT_EQ(results[3], DevalF(y, x4));
    ASSERT_EQ(results[4], DevalF(y, x5));
}

// Reverse AD test
TEST(OpsTest, BasicBinaryOps3)
{
    Variable x1{1}, x2{2}, x3{-3}, x4{24}, x5{-99};
    Expression y = 9 + (x1 + x2) * (x3 + x4 * x5) * 3 - 2 * x1 * 4 - 10;

    Type results[5] = {-7145, -7137, 9, -891, 216};

    PreComp(y);
    ASSERT_EQ(results[0], DevalR(y, x1));
    ASSERT_EQ(results[1], DevalR(y, x2));
    ASSERT_EQ(results[2], DevalR(y, x3));
    ASSERT_EQ(results[3], DevalR(y, x4));
    ASSERT_EQ(results[4], DevalR(y, x5));
}

// Jacobian eval test
TEST(OpsTest, BasicBinaryOps4)
{
    Variable x1{1}, x2{2}, x3{-3}, x4{24}, x5{-99};
    Expression y = 9 + (x1 + x2) * (x3 + x4 * x5) * 3 - 2 * x1 * 4 - 10;

    Variable tmp[5] = {x1, x2, x3, x4, x5};
    Expression DY[5];

    Type results[5] = {-7145, -7137, 9, -891, 216};

    // Jacobian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, tmp[i]);
        ASSERT_EQ(results[i], Eval(DY[i]));
    }
}

// Hessian eval test
TEST(OpsTest, BasicBinaryOps5)
{
    Variable x1{1}, x2{2}, x3{-3}, x4{24}, x5{-99};
    Expression y = 9 + (x1 + x2) * (x3 + x4 * x5) * 3 - 2 * x1 * 4 - 10;

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5][5];

    Type results[5][5] = {{0, 0, 3, -297, 72},
                          {0, 0, 3, -297, 72},
                          {3, 3, 0, 0, 0},
                          {-297, -297, 0, 0, 9},
                          {72, 72, 0, 9, 0}};

    // Hessian computation
    for (int i{}; i < 5; ++i)
    {
        for (int j{}; j < 5; ++j)
        {
            DY[i][j] = SymDiff(SymDiff(y, *tmp[i]), *tmp[j]);
            ASSERT_EQ(results[i][j], Eval(DY[i][j]));
        }
    }
}

// Jacobian based hessian Forward test
TEST(OpsTest, BasicBinaryOps6)
{
    Variable x1{1}, x2{2}, x3{-3}, x4{24}, x5{-99};
    Expression y = 9 + (x1 + x2) * (x3 + x4 * x5) * 3 - 2 * x1 * 4 - 10;

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type results[5][5] = {{0, 0, 3, -297, 72},
                          {0, 0, 3, -297, 72},
                          {3, 3, 0, 0, 0},
                          {-297, -297, 0, 0, 9},
                          {72, 72, 0, 9, 0}};

    // Jacobian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        for (int j{}; j < 5; ++j)
        {
            ASSERT_EQ(results[i][j], DevalF(DY[i], *tmp[j]));
        }
    }
}

// Jacobian based hessian Reverse test
TEST(OpsTest, BasicBinaryOps7)
{
    Variable x1{1}, x2{2}, x3{-3}, x4{24}, x5{-99};
    Expression y = 9 + (x1 + x2) * (x3 + x4 * x5) * 3 - 2 * x1 * 4 - 10;

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type results[5][5] = {{0, 0, 3, -297, 72},
                          {0, 0, 3, -297, 72},
                          {3, 3, 0, 0, 0},
                          {-297, -297, 0, 0, 9},
                          {72, 72, 0, 9, 0}};

    // Jacobian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
            ASSERT_EQ(results[i][j], DevalR(DY[i], *tmp[j]));
        }
    }
}

// Hessian split test (Recurive expression)
TEST(OpsTest, BasicBinaryOps8)
{
    Variable x1{1}, x2{2}, x3{-3}, x4{24}, x5{-99};

    // Split test
    Expression y = 9 + (x1 + x2);
    y = (y - 9) * (x3 + x4 * x5) * 3 + 9;
    y = y - 2 * x1 * 4 - 10;

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5][5];

    Type results[5][5] = {{0, 0, 3, -297, 72},
                          {0, 0, 3, -297, 72},
                          {3, 3, 0, 0, 0},
                          {-297, -297, 0, 0, 9},
                          {72, 72, 0, 9, 0}};

    // Hessian computation
    for (int i{}; i < 5; ++i)
    {
        for (int j{}; j < 5; ++j)
        {
            DY[i][j] = SymDiff(SymDiff(y, *tmp[i]), *tmp[j]);
            ASSERT_EQ(results[i][j], Eval(DY[i][j]));
        }
    }
}

// Sine-cosine jacobian test
TEST(OpsTest, SineCosineOps)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{2}, x3{6.2}, x4{13}, x5{108};

    // Split test
    Expression y = sin(x1) * cos(x2) * (x3 + sin(x4 * x5)) / cos(x1);

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = 6.9445;
    Type results[5] = {-20.5621, 15.1740, 1.0704, -110.7118, -13.3264};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Jacobian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
#if defined(USE_COMPLEX_MATH)
        ASSERT_NEAR(results[i].real(), Eval(DY[i]).real(), epi);
        ASSERT_NEAR(results[i].imag(), Eval(DY[i]).imag(), epi);
#else
        ASSERT_NEAR(results[i], Eval(DY[i]), epi);
#endif
    }
}

// Tan jacobian test
TEST(OpsTest, TanOps)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{2}, x3{6.2}, x4{13}, x5{108};

    // Split test
    Expression y =
        10 - (tan(x1) * cos(x2 * 2 + 3) * (x3 + tan(x4 * x5)) / cos(x1)) - x2;

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = 39.571;
    Type results[5] = {-174.6853, -56.0251, 5.3515, 630.1502, 75.8514};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Jacobian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
#if defined(USE_COMPLEX_MATH)
        ASSERT_NEAR(results[i].real(), Eval(DY[i]).real(), epi);
        ASSERT_NEAR(results[i].imag(), Eval(DY[i]).imag(), epi);
#else
        ASSERT_NEAR(results[i], Eval(DY[i]), epi);
#endif
    }
}

// Tan jacobian test
TEST(OpsTest, SqrtOps)
{
    double epi{0.001};
    Variable x1{1.5}, x2{6.8}, x3{8.5}, x4{1}, x5{10.8};

    // Split test
    Expression y = 5 * sqrt(x1) * sqrt(x5 + x2);
    y = tan(x1) * (y + sqrt(x4 * x3)) * cos(x2);

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = 350.701;
    Type results[5] = {5075.24343, -190.36993, 2.10253, 17.87149, 8.94768};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Jacobian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
#if defined(USE_COMPLEX_MATH)
        ASSERT_NEAR(results[i].real(), Eval(DY[i]).real(), epi);
        ASSERT_NEAR(results[i].imag(), Eval(DY[i]).imag(), epi);
#else
        ASSERT_NEAR(results[i], Eval(DY[i]), epi);
#endif
    }
}

// Tan Hessian test
TEST(OpsTest, SqrtOpsHessian)
{
    double epi{0.001};
    Variable x1{1.5}, x2{6.8}, x3{8.5}, x4{1}, x5{10.8};

    // Split test
    Expression y = 5 * sqrt(x1) * sqrt(x5 + x2);
    y = tan(x1) * (y + sqrt(x4 * x3)) * cos(x2);

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = 350.701;
    Type results[5][5] = {
        {143116.1727, -2754.671641, 29.79772948, 253.2807006, 129.7921031},
        {-2754.671641, -361.1262211, -1.194950669, -10.15708068, -5.339520809},
        {29.79772948, -1.194950669, -0.1236781125, 1.051263956, 0},
        {253.2807006, -10.15708068, 1.051263956, -8.935743626, 0},
        {129.7921031, -5.339520809, 0, 0, -0.2541955066}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Hessian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
#if defined(USE_COMPLEX_MATH)
            // DevalF
            ASSERT_NEAR(results[i][j].real(),
                        DevalF(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalF(DY[i], *tmp[j]).imag(),
                        epi);
            // DevalR
            ASSERT_NEAR(results[i][j].real(),
                        DevalR(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalR(DY[i], *tmp[j]).imag(),
                        epi);
#else
            ASSERT_NEAR(results[i][j], DevalF(DY[i], *tmp[j]), epi);
            ASSERT_NEAR(results[i][j], DevalR(DY[i], *tmp[j]), epi);
#endif
        }
    }
}

// Exp jacobian test
TEST(OpsTest, ExpOps)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{2}, x3{6.2}, x4{13}, x5{108};

    // Split test
    Expression y = exp(((x1 - 0.35) * (x2 - x3)) / (2 * 10));
    y = x2 * y * x4;

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = 36.003;
    Type results[5] = {-7.5606, 15.2112, 2.7902, 2.7694, 0};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Jacobian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
#if defined(USE_COMPLEX_MATH)
        ASSERT_NEAR(results[i].real(), Eval(DY[i]).real(), epi);
        ASSERT_NEAR(results[i].imag(), Eval(DY[i]).imag(), epi);
#else
        ASSERT_NEAR(results[i], Eval(DY[i]), epi);
#endif
    }
}

// Exp Hessian test
TEST(OpsTest, ExpOpsHessian)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{2}, x3{6.2}, x4{13}, x5{108};

    // Split test
    Expression y = exp(((x1 - 0.35) * (x2 - x3)) / (2 * 10));
    y = x2 * y * x4;

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = 36.003;
    Type results[5][5] = {{1.5877, -1.3942, -2.3861, -0.5816, 0},
                          {-1.3942, -2.5740, 1.1789, 1.1701, 0},
                          {-2.3861, 1.1789, 0.2162, 0.2146, 0},
                          {-0.5816, 1.1701, 0.2146, 0, 0},
                          {0, 0, 0, 0, 0}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Hessian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
#if defined(USE_COMPLEX_MATH)
            // DevalF
            ASSERT_NEAR(results[i][j].real(),
                        DevalF(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalF(DY[i], *tmp[j]).imag(),
                        epi);
            // DevalR
            ASSERT_NEAR(results[i][j].real(),
                        DevalR(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalR(DY[i], *tmp[j]).imag(),
                        epi);
#else
            ASSERT_NEAR(results[i][j], DevalF(DY[i], *tmp[j]), epi);
            ASSERT_NEAR(results[i][j], DevalR(DY[i], *tmp[j]), epi);
#endif
        }
    }
}

// Log jacobian test
TEST(OpsTest, LogOps)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{2}, x3{6.2}, x4{13}, x5{108};

    // Split test
    Expression y =
        2.3 + log(-1 * x1 * x2) * exp(log(x3 * x5)) - sin(x2) * tan(x4);
    y = x2 * y * x4 * cos(x3);

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = 15237.54114;
    Type results[5] = {-14457.8327436,
                       16298.4626970,
                       3720.2869560,
                       1143.5075761,
                       140.6375620};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Jacobian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
#if defined(USE_COMPLEX_MATH)
        ASSERT_NEAR(results[i].real(), Eval(DY[i]).real(), epi);
        ASSERT_NEAR(results[i].imag(), Eval(DY[i]).imag(), epi);
#else
        ASSERT_NEAR(results[i], Eval(DY[i]), epi);
#endif
    }
}

// Log Hessian test
TEST(OpsTest, LogOpsHessian)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{2}, x3{6.2}, x4{13}, x5{108};

    // Split test
    Expression y =
        2.3 + log(-1 * x1 * x2) * exp(log(x3 * x5)) - sin(x2) * tan(x4);
    y = x2 * y * x4 * cos(x3);

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = 15237.54114;
    Type results[5][5] = {
        {-12048.1939530,
         -7228.9163718,
         -3537.3695631,
         -1112.1409803,
         -133.8688217},
        {-7228.9163718, 4353.2510736, 3982.9814775, 1252.5164412, 150.6400740},
        {-3537.36956311,
         3982.98147749,
         -14829.02108763,
         283.79040213,
         34.40951628},
        {-1112.14098028,
         1252.51644120,
         283.79040213,
         -30.89665861,
         10.81827400},
        {-133.868821700, 150.640074003, 34.409516277, 10.818273997, 0}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Hessian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
#if defined(USE_COMPLEX_MATH)
            // DevalF
            ASSERT_NEAR(results[i][j].real(),
                        DevalF(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalF(DY[i], *tmp[j]).imag(),
                        epi);
            // DevalR
            ASSERT_NEAR(results[i][j].real(),
                        DevalR(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalR(DY[i], *tmp[j]).imag(),
                        epi);
#else
            ASSERT_NEAR(results[i][j], DevalF(DY[i], *tmp[j]), epi);
            ASSERT_NEAR(results[i][j], DevalR(DY[i], *tmp[j]), epi);
#endif
        }
    }
}

// Asin jacobian test
TEST(OpsTest, ASinOps)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{0.1}, x3{6.2}, x4{13}, x5{108};

    // Split test
    Type a{-0.707, 0.5};
    Expression y = x4 * x5 * asin(a * x2 * x3);
    y = x2 * y * tan(x4) * cos(x3) + y * sin(sqrt(x1) * exp(x2 * x3));

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value{-1796.097735, -2240.624749};
    Type results[5] = {{1552.43028979, 1986.02631938},
                       {-43799.27148183, -50235.51540389},
                       {-704.28029998, -811.93832801},
                       {-210.85810181, -115.53041168},
                       {-16.63053458, -20.74652546}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Jacobian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
#if defined(USE_COMPLEX_MATH)
        ASSERT_NEAR(results[i].real(), Eval(DY[i]).real(), epi);
        ASSERT_NEAR(results[i].imag(), Eval(DY[i]).imag(), epi);
#else
        ASSERT_NEAR(results[i], Eval(DY[i]), epi);
#endif
    }
}

// Asin Hessian test
TEST(OpsTest, ASinOpsHessian)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{0.1}, x3{2.2}, x4{0.13}, x5{1.8};
    Type a{-0.707, 0.5};

    // Split test
    Expression y = x4 * x5 * asin(a * x2 * x3);
    y = x2 * y * tan(x4) * cos(x3) + y * sin(sqrt(x1) * exp(x2 * x3));

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = {-0.04730483617, -0.06666974192};
    Type results[5][5] = {{{-0.002542547, -0.003551643},
                           {0.463684194, 0.637152051},
                           {0.021076554, 0.028961457},
                           {0.237231576, 0.331385001},
                           {0.017133392, 0.023933361}},
                          {{0.463684194, 0.637152051},
                           {-4.339162697, -5.573482702},
                           {-0.484204319, -0.663048577},
                           {-4.899775560, -6.902101430},
                           {-0.357005608, -0.496214752}},
                          {{0.021076554, 0.028961457},
                           {-0.484204319, -0.663048577},
                           {-0.009012180, -0.011485935},
                           {-0.218742714, -0.316577042},
                           {-0.016084819, -0.022657378}},
                          {{0.237231576, 0.331385001},
                           {-4.899775560, -6.902101430},
                           {-0.218742714, -0.316577042},
                           {0.034012707, -0.024348984},
                           {-0.200949709, -0.285778002}},
                          {{0.017133392, 0.023933361},
                           {-0.357005608, -0.496214752},
                           {-0.016084819, -0.022657378},
                           {-0.200949709, -0.285778002},
                           {0, 0}}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Hessian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
#if defined(USE_COMPLEX_MATH)
            // DevalF
            ASSERT_NEAR(results[i][j].real(),
                        DevalF(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalF(DY[i], *tmp[j]).imag(),
                        epi);
            // DevalR
            ASSERT_NEAR(results[i][j].real(),
                        DevalR(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalR(DY[i], *tmp[j]).imag(),
                        epi);
#else
            ASSERT_NEAR(results[i][j], DevalF(DY[i], *tmp[j]), epi);
            ASSERT_NEAR(results[i][j], DevalR(DY[i], *tmp[j]), epi);
#endif
        }
    }
}

// Tanh jacobian test
TEST(OpsTest, TanhOps)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{0.1}, x3{6.2}, x4{13}, x5{108};

    // Split test
    Type a{-0.707, 0.5};
    Expression y = x4 * x5 * tanh(a * x2 * x3);
    y = x2 * y * tan(x4) * cos(x3) + tanh(y * 2) * sin(sqrt(x1) * exp(x2 * x3));

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value{-28.94381007, 13.16350226};
    Type results[5] = {{0, 3.306156431},
                       {-583.013791417, 228.245717412},
                       {-7.148360529, 2.362356436},
                       {-78.138818106, 45.704271223},
                       {-0.267998241, 0.156755178}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Jacobian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
#if defined(USE_COMPLEX_MATH)
        ASSERT_NEAR(results[i].real(), Eval(DY[i]).real(), epi);
        ASSERT_NEAR(results[i].imag(), Eval(DY[i]).imag(), epi);
#else
        ASSERT_NEAR(results[i], Eval(DY[i]), epi);
#endif
    }
}

// Tanh Hessian test
TEST(OpsTest, TanhOpsHessian)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{1.4}, x3{0.02}, x4{-2.2}, x5{1.8};

    // Split test
    Type a{2, 5};
    Expression y = x4 * x5 * tanh(a * x2 * x3);
    y = sqrt(x2) * y * cos(a * x1);

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = {-49.91794653, 134.32152261};
    Type results[5][5] = {{{1638.153575040, 3819.110905412},
                           {-26.402783132, -832.453449174},
                           {-1372.098603744, -38985.668696854},
                           {8.656294827, 350.655868097},
                           {-10.579915900, -428.579394340}},
                          {{-26.402783132, -832.453449174},
                           {-17.272413136, 55.506188183},
                           {-2530.485327060, 7519.136714711},
                           {24.025752865, -66.067337125},
                           {-29.364809057, 80.748967597}},
                          {{-1372.098603744, -38985.668696854},
                           {-2530.485327060, 7519.136714711},
                           {6767.322949578, 15114.681126824},
                           {1114.553308144, -3098.332660018},
                           {-1362.231821065, 3786.851028910}},
                          {{8.656294827, 350.655868097},
                           {24.025752865, -66.067337125},
                           {1114.553308144, -3098.332660018},
                           {0, 0},
                           {12.605542053, -33.919576416}},
                          {{-10.579915900, -428.579394340},
                           {-29.364809057, 80.748967597},
                           {-1362.231821065, 3786.851028910},
                           {12.605542053, -33.919576416},
                           {0, 0}}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Hessian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
#if defined(USE_COMPLEX_MATH)
            // DevalF
            ASSERT_NEAR(results[i][j].real(),
                        DevalF(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalF(DY[i], *tmp[j]).imag(),
                        epi);
            // DevalR
            ASSERT_NEAR(results[i][j].real(),
                        DevalR(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalR(DY[i], *tmp[j]).imag(),
                        epi);
#else
            ASSERT_NEAR(results[i][j], DevalF(DY[i], *tmp[j]), epi);
            ASSERT_NEAR(results[i][j], DevalR(DY[i], *tmp[j]), epi);
#endif
        }
    }
}

// ACos Hessian test
TEST(OpsTest, ACosOpsHessian)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{1.4}, x3{0.02}, x4{-2.2}, x5{1.8};

    // Split test
    Type a{2, 5};
    Expression y = acos(x4) * sin(x5) * tanh(a * x2 * x3);
    y = sqrt(x2) * acos(y) * cos(a * x1);

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = {-273.165517466639, -131.925278583887};
    Type results[5][5] = {{{-8374.98143847715, 2692.87949907115},
                           {429.127007787236, -402.037481033474},
                           {-10702.9845856282, -30975.5033100249},
                           {95.3816205191256, 9.93232498454368},
                           {47.8919769751865, 143.466600415969}},
                          {{429.127007787236, -402.037481033474},
                           {18.2342113246382, 77.1719079887916},
                           {-1266.44923418445, 6395.11178024989},
                           {-15.0202783718285, -11.376258755738},
                           {6.55111601203226, -28.9867963649231}},
                          {{-10702.9845856282, -30975.5033100249},
                           {-1266.44923418445, 6395.11178024989},
                           {-66839.5459919442, -8248.49914453643},
                           {-657.413548845632, -589.079897690757},
                           {417.661385167509, -1328.11159779581}},
                          {{95.3816205191256, 9.93232498454368},
                           {-15.0202783718285, -11.376258755738},
                           {-657.413548845632, -589.079897690757},
                           {-8.73396730083714, -4.14807854757945},
                           {3.0053353606285, 2.75038351878748}},
                          {{47.8919769751865, 143.466600415969},
                           {6.55111601203226, -28.9867963649231},
                           {417.661385167509, -1328.11159779581},
                           {3.0053353606285, 2.75038351878748},
                           {5.40522630778999, -120.613299842311}}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Hessian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
#if defined(USE_COMPLEX_MATH)
            // DevalF
            ASSERT_NEAR(results[i][j].real(),
                        DevalF(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalF(DY[i], *tmp[j]).imag(),
                        epi);
            // DevalR
            ASSERT_NEAR(results[i][j].real(),
                        DevalR(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalR(DY[i], *tmp[j]).imag(),
                        epi);
#else
            ASSERT_NEAR(results[i][j], DevalF(DY[i], *tmp[j]), epi);
            ASSERT_NEAR(results[i][j], DevalR(DY[i], *tmp[j]), epi);
#endif
        }
    }
}

// ATan Hessian test
TEST(OpsTest, ATanOpsHessian)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{1.4}, x3{0.02}, x4{-2.2}, x5{1.8};

    // Split test
    Type a{-1, 3};
    Expression y =
        acos(x4) * sin(atan(x5 * x1)) * tan(100 * cosh(x2) * (pow(x3, a)));
    y = 10 + 100 * (pow(10.352, cosh(y))) * cos(pow(x2, sinh(2 * x3)));

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = {10.2806874164658, 0.593755259685568};
    Type results[5][5] = {{{-2.32033536828771, 3.18716263380208},
                           {0.0202361439883069, -0.0524088588938473},
                           {0.476877215829783, -1.23504708844508},
                           {0.0761919987870955, 1.53662483428487},
                           {1.30504966612832, -1.49844090183159}},
                          {{0.0202361439883069, -0.0524088588938473},
                           {0.00871169381708299, 0.0184283787630819},
                           {-0.667093216970284, -1.41114308316308},
                           {-0.039588596407824, -0.0403103479498996},
                           {-0.0134907626588713, 0.0349392392625649}},
                          {{0.476877215829783, -1.23504708844508},
                           {-0.667093216970284, -1.41114308316308},
                           {-0.362462127830519, -0.766738308207669},
                           {-0.932929694732402, -0.949938215036703},
                           {-0.317918143886523, 0.823364725630056}},
                          {{0.0761919987870955, 1.53662483428487},
                           {-0.039588596407824, -0.0403103479498996},
                           {-0.932929694732402, -0.949938215036703},
                           {2.09570464571657, 1.00382058876039},
                           {-0.0507946658580632, -1.02441655618991}},
                          {{1.30504966612832, -1.49844090183159},
                           {-0.0134907626588713, 0.0349392392625649},
                           {-0.317918143886523, 0.823364725630056},
                           {-0.0507946658580633, -1.02441655618991},
                           {-1.03126016368343, 1.41651672613426}}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Hessian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
#if defined(USE_COMPLEX_MATH)
            // DevalF
            ASSERT_NEAR(results[i][j].real(),
                        DevalF(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalF(DY[i], *tmp[j]).imag(),
                        epi);
            // DevalR
            ASSERT_NEAR(results[i][j].real(),
                        DevalR(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalR(DY[i], *tmp[j]).imag(),
                        epi);
#else
            ASSERT_NEAR(results[i][j], DevalF(DY[i], *tmp[j]), epi);
            ASSERT_NEAR(results[i][j], DevalR(DY[i], *tmp[j]), epi);
#endif
        }
    }
}

// Pow Hessian test
TEST(OpsTest, PowOpsHessian)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{1.4}, x3{0.02}, x4{-2.2}, x5{1.8};

    // Split test
    Type a{2, 5};
    Expression y = acos(x4) * sin(x5) * tan(x2 * pow(x3, a));
    y = 10 + 100 * (pow(0.1, y)) * cos(pow(x2, x3));

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = {63.3630594727319, 0.209712914306739};
    Type results[5][5] = {{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
                          {{0, 0},
                           {0.841069758381305, -0.00389596087675679},
                           {-104.174587150829, -2.97377411481931},
                           {-0.0152522166776553, -0.0180141708500201},
                           {0.0159338180771647, -0.0336456622675089}},
                          {{0, 0},
                           {-104.174587150829, -2.97377411481931},
                           {-2086.52518071918, -15703.0948925074},
                           {4.31782144254902, -8.10578499968465},
                           {14.4559169291753, 0.919257081475748}},
                          {{0, 0},
                           {-0.0152522166776553, -0.0180141708500201},
                           {4.31782144254902, -8.10578499968465},
                           {-0.0127235128138507, -0.0148786482343272},
                           {0.00514662525459841, 0.00607700858246767}},
                          {{0, 0},
                           {0.0159338180771647, -0.0336456622675089},
                           {14.4559169291753, 0.919257081475748},
                           {0.00514662525459841, 0.00607700858246767},
                           {0.0980585060817473, -0.209371324644246}}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Hessian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
#if defined(USE_COMPLEX_MATH)
            // DevalF
            ASSERT_NEAR(results[i][j].real(),
                        DevalF(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalF(DY[i], *tmp[j]).imag(),
                        epi);
            // DevalR
            ASSERT_NEAR(results[i][j].real(),
                        DevalR(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalR(DY[i], *tmp[j]).imag(),
                        epi);
#else
            ASSERT_NEAR(results[i][j], DevalF(DY[i], *tmp[j]), epi);
            ASSERT_NEAR(results[i][j], DevalR(DY[i], *tmp[j]), epi);
#endif
        }
    }
}

// Sinh Hessian test
TEST(OpsTest, SinhOpsHessian)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{1.4}, x3{0.02}, x4{-2.2}, x5{1.8};

    // Split test
    Type a{2, 5};
    Expression y = acos(x4) * sin(sinh(x5)) * tan(x2 * pow(x3, a));
    y = 10 + 100 * (pow(0.1, y)) * cos(pow(x2, sinh(2 * x3)));

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = {62.8651923799783, 0.0422614079926355};
    Type results[5][5] = {{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
                          {{0, 0},
                           {1.64206717051515, -0.00150982944676913},
                           {-133.986695184988, -0.597460060164591},
                           {-0.00299466012675725, -0.00349180276448447},
                           {0.201905577486137, -0.433426133285376}},
                          {{0, 0},
                           {-133.986695184988, -0.597460060164591},
                           {-493.034482377988, -3163.69882720286},
                           {0.863618006436983, -1.63833898252752},
                           {192.168420489359, 11.3836522999473}},
                          {{0, 0},
                           {-0.00299466012675725, -0.00349180276448447},
                           {0.863618006436983, -1.63833898252752},
                           {-0.00257288607479395, -0.00299377865960351},
                           {0.0689560062452508, 0.0803943475451993}},
                          {{0, 0},
                           {0.201905577486137, -0.433426133285376},
                           {192.168420489359, 11.3836522999473},
                           {0.0689560062452508, 0.0803943475451993},
                           {0.469196022297281, -1.03035522446664}}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Hessian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
#if defined(USE_COMPLEX_MATH)
            // DevalF
            ASSERT_NEAR(results[i][j].real(),
                        DevalF(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalF(DY[i], *tmp[j]).imag(),
                        epi);
            // DevalR
            ASSERT_NEAR(results[i][j].real(),
                        DevalR(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalR(DY[i], *tmp[j]).imag(),
                        epi);
#else
            ASSERT_NEAR(results[i][j], DevalF(DY[i], *tmp[j]), epi);
            ASSERT_NEAR(results[i][j], DevalR(DY[i], *tmp[j]), epi);
#endif
        }
    }
}

// Cosh Hessian test
TEST(OpsTest, CoshOpsHessian)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{1.4}, x3{0.02}, x4{-2.2}, x5{1.8};

    // Split test
    Type a{2, 5};
    Expression y =
        acos(x4) * sin(sinh(x5)) * tan(100 * cosh(x2) * (pow(x3, a)));
    y = 10 + 100 * (pow(10.352, cosh(y))) * cos(pow(x2, sinh(2 * x3)));

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = {556.024929358879, -1.67768449736836};
    Type results[5][5] = {{{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}},
                          {{0, 0},
                           {11.9549073291295, -5.65587920271092},
                           {-383.839285374228, -1826.29437145167},
                           {1.07509503212658, -0.324678208021175},
                           {76.9053790161931, 87.8197182960227}},
                          {{0, 0},
                           {-383.839285374228, -1826.29437145167},
                           {608298.048028554, 106950.932805879},
                           {218.12221627133, 274.369386569261},
                           {-16602.8010573191, 32430.0127098664}},
                          {{0, 0},
                           {1.07509503212658, -0.324678208021175},
                           {218.12221627133, 274.369386569261},
                           {0.346135188727419, -0.00990896823779426},
                           {-19.1906018205607, 5.68874927472991}},
                          {{0, 0},
                           {76.9053790161931, 87.8197182960227},
                           {-16602.8010573191, 32430.0127098664},
                           {-19.1906018205607, 5.68874927472991},
                           {-612.354185354542, -700.32531577533}}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Hessian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
#if defined(USE_COMPLEX_MATH)
            // DevalF
            ASSERT_NEAR(results[i][j].real(),
                        DevalF(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalF(DY[i], *tmp[j]).imag(),
                        epi);
            // DevalR
            ASSERT_NEAR(results[i][j].real(),
                        DevalR(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalR(DY[i], *tmp[j]).imag(),
                        epi);
#else
            ASSERT_NEAR(results[i][j], DevalF(DY[i], *tmp[j]), epi);
            ASSERT_NEAR(results[i][j], DevalR(DY[i], *tmp[j]), epi);
#endif
        }
    }
}

// ACosh & ASinh Hessian test
TEST(OpsTest, ASinhACoshOpsHessian)
{
    double epi{0.001};
    Variable x1{-1.2}, x2{1.4}, x3{0.02}, x4{-2.2}, x5{1.8};

    // Split test
    Type a{-1, 3};
    Expression y = exp((x1 - 2) / (10 * x3)) * 2000 * asinh(acosh(x1 * x2));
    y = 100 * (pow(10.352, acosh(sin(sin(x4) * y)))) +
        cos(pow(x2, sinh(2 * asinh(a * x3)))) - exp(x5) / 10;

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = {-86.3555252808744, 50.5842800673248};
    Type results[5][5] = {
        {{-0.0447333501975899, -2.35728638100384},
         {0.0624155468393745, -0.00641069298088393},
         {5.24745699274713, -354.867124360021},
         {0.00573200605296676, -0.344527215876792},
         {0, 0}},
        {{0.0624155468393745, -0.00641069298088394},
         {-0.023534836995084, 0.0550017282648419},
         {10.8767022293426, -4.76603099524123},
         {0.00844873233948765, -0.00136387343252629},
         {0, 0}},
        {{5.24745699274724, -354.86712436002},
         {10.8767022293426, -4.76603099524123},
         {2408.36464088845, -53247.9979432948},
         {2.49421767184571, -55.3789442476917},
         {0, 0}},
        {{0.00573200605296679, -0.344527215876792},
         {0.00844873233948765, -0.00136387343252629},
         {2.49421767184571, -55.3789442476917},
         {-0.00417027911567986, 0.0951815566717442},
         {0, 0}},
        {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {-0.604964746441295, 0}}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Hessian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
#if defined(USE_COMPLEX_MATH)
            // DevalF
            ASSERT_NEAR(results[i][j].real(),
                        DevalF(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalF(DY[i], *tmp[j]).imag(),
                        epi);
            // DevalR
            ASSERT_NEAR(results[i][j].real(),
                        DevalR(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalR(DY[i], *tmp[j]).imag(),
                        epi);
#else
            ASSERT_NEAR(results[i][j], DevalF(DY[i], *tmp[j]), epi);
            ASSERT_NEAR(results[i][j], DevalR(DY[i], *tmp[j]), epi);
#endif
        }
    }
}

// ATanh Hessian test
TEST(OpsTest, ATanhOpsHessian)
{
    double epi{1};
    Variable x1{-1.2}, x2{1.4}, x3{0.02}, x4{-2.2}, x5{1.8};

    // Split test
    Type a{-6.2, 3.25};
    Expression y = exp((x1 - 2) / (10 * x3 * atanh(x5))) * 2000 *
                   (a * asin(0.01 * acosh(x1 * x2))) * x5;
    y = 100 * (pow(10.352, acosh(sin(sin(x4) * y)))) +
        cos(pow(x2, sinh(2 * asinh(a * x3)))) - exp(x5 * x5) / 10;

    Variable *tmp[5] = {&x1, &x2, &x3, &x4, &x5};
    Expression DY[5];

    Type value = {15985.1085706817, -13837.5948413062};
    Type results[5][5] = {{{497657510.919682, 122726883.274175},
                           {-41041990.8527138, -8595889.86216499},
                           {71778215979.803, 18075291268.0761},
                           {15204334.7772191, 113077584.844619},
                           {-217414732.245122, 402974430.841756}},
                          {{-41041990.8527138, -8595889.86216499},
                           {3646086.12044203, 570845.087204471},
                           {-5920754310.05371, -1259515328.38437},
                           {-1574264.62223299, -9179890.52645786},
                           {16554398.9632036, -33580126.6957005}},
                          {{71778215979.803, 18075291268.0761},
                           {-5920754310.05371, -1259515328.38437},
                           {10356042683657.7, 2662175771029.02},
                           {2138830834.87156, 16378834010.2002},
                           {-31696202686.0881, 58207618618.1501}},
                          {{15204334.7772191, 113077584.844619},
                           {-1574264.62223299, -9179890.52645786},
                           {2138830834.87156, 16378834010.2002},
                           {-23062037.6801985, 11119435.5737655},
                           {-101444965.3059, -12084280.1352799}},
                          {{-217414732.245122, 402974430.841756},
                           {16554398.9632036, -33580126.6957005},
                           {-31696202686.0881, 58207618618.1501},
                           {-101444965.3059, -12084280.1352799},
                           {-299872860.614912, -282416374.543831}}};

// Expression evaluation
#if defined(USE_COMPLEX_MATH)
    ASSERT_NEAR(value.real(), Eval(y).real(), epi);
    ASSERT_NEAR(value.imag(), Eval(y).imag(), epi);
    //std::cout << Eval(y).real() << " " << Eval(y).imag() << "\n";
#else
    ASSERT_NEAR(value, Eval(y), epi);
#endif

    // Hessian evaluation
    for (int i{}; i < 5; ++i)
    {
        DY[i] = SymDiff(y, *tmp[i]);
        PreComp(DY[i]);
        for (int j{}; j < 5; ++j)
        {
#if defined(USE_COMPLEX_MATH)
            // DevalF
            ASSERT_NEAR(results[i][j].real(),
                        DevalF(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalF(DY[i], *tmp[j]).imag(),
                        epi);
            //std::cout << DevalF(DY[i],*tmp[j]).real() << " " << DevalF(DY[i],*tmp[j]).imag() << "\n";

            // DevalR
            ASSERT_NEAR(results[i][j].real(),
                        DevalR(DY[i], *tmp[j]).real(),
                        epi);
            ASSERT_NEAR(results[i][j].imag(),
                        DevalR(DY[i], *tmp[j]).imag(),
                        epi);
#else
            ASSERT_NEAR(results[i][j], DevalF(DY[i], *tmp[j]), epi);
            ASSERT_NEAR(results[i][j], DevalR(DY[i], *tmp[j]), epi);
#endif
        }
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}