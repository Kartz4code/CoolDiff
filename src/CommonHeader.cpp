/**
 * @file src/CommonHeader.cpp
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

#include "CommonHeader.hpp"
#include <charconv>

#if defined(USE_COMPLEX_MATH) 
    // Addition (Type and complex number)
    Type operator+(Real val, const Type& cmx) {
        return {val + cmx.real(), cmx.imag()};
    }

    Type operator+(const Type& cmx, Real val) {
        return {val + cmx.real(), cmx.imag()};
    }

    // Subtraction (Type and complex number)
    Type operator-(Real val, const Type& cmx) {
        return {val - cmx.real(), cmx.imag()};
    }

    Type operator-(const Type& cmx, Real val) {
        return {cmx.real() - val, cmx.imag()};
    }

    // Multiplication (Type and complex number)
    Type operator*(Real val, const Type& cmx) {
        return {val*cmx.real(), val*cmx.imag()};
    }

    Type operator*(const Type& cmx, Real val) {
        return {val*cmx.real(), val*cmx.imag()};
    }

    // Division (Type and complex number)
    Type operator/(Real val, const Type& cmx) {
        Real abs_sq = (cmx*cmx).real();
        return {(val*cmx.real())/abs_sq, (-1*val*cmx.imag())/abs_sq};
    }

    Type operator/(const Type& cmx, Real val) {
        return {cmx.real()/val, cmx.imag()/val};
    }

    // Not equal (Type and complex number)
    bool operator!=(const Type& cmx, Real val) {
        return !(cmx.real() == val && cmx.imag() == val);
    }

    bool operator!=(Real val, const Type& cmx) {
        return !(cmx.real() == val && cmx.imag() == val);
    }

    // Equal (Type and complex number)
    bool operator==(const Type& cmx, Real val) {
        return (cmx.real() == val && cmx.imag() == val);
    }

    bool operator==(Real val, const Type& cmx) {
        return (cmx.real() == val && cmx.imag() == val);
    }

#endif