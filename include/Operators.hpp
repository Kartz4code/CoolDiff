/**
 * @file include/Operators.hpp
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

#pragma once

#include <cmath>

template <typename T>
struct Sin
{
    T operator()(T x) const
    {
        return std::sin(x);
    }
};

template <typename T>
struct Cos
{
    T operator()(T x) const
    {
        return std::cos(x);
    }
};

template <typename T>
struct Tan
{
    T operator()(T x) const
    {
        return std::tan(x);
    }
};

template <typename T>
struct Sinh
{
    T operator()(T x) const
    {
        return std::sinh(x);
    }
};

template <typename T>
struct Cosh
{
    T operator()(T x) const
    {
        return std::cosh(x);
    }
};

template <typename T>
struct Tanh
{
    T operator()(T x) const
    {
        return std::tanh(x);
    }
};

template <typename T>
struct ASin
{
    T operator()(T x) const
    {
        return std::asin(x);
    }
};

template <typename T>
struct ACos
{
    T operator()(T x) const
    {
        return std::acos(x);
    }
};

template <typename T>
struct ATan
{
    T operator()(T x) const
    {
        return std::atan(x);
    }
};

template <typename T>
struct ASinh
{
    T operator()(T x) const
    {
        return std::asinh(x);
    }
};

template <typename T>
struct ACosh
{
    T operator()(T x) const
    {
        return std::acosh(x);
    }
};

template <typename T>
struct ATanh
{
    T operator()(T x) const
    {
        return std::atanh(x);
    }
};
