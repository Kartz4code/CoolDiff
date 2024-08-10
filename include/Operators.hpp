#pragma once

template<typename T>
struct Sin {
    T operator()(T x) const {
        return std::sin(x);
    }
};

template<typename T>
struct Cos {
    T operator()(T x) const {
        return std::cos(x);
    }
};

template<typename T>
struct Tan {
    T operator()(T x) const {
        return std::tan(x);
    }
};

template<typename T>
struct Sinh {
    T operator()(T x) const {
        return std::sinh(x);
    }
};

template<typename T>
struct Cosh {
    T operator()(T x) const {
        return std::cosh(x);
    }
};

template<typename T>
struct Tanh {
    T operator()(T x) const {
        return std::tanh(x);
    }
};

template<typename T>
struct ASin {
    T operator()(T x) const {
        return std::asin(x);
    }
};

template<typename T>
struct ACos {
    T operator()(T x) const {
        return std::acos(x);
    }
};

template<typename T>
struct ATan {
    T operator()(T x) const {
        return std::atan(x);
    }
};

template<typename T>
struct ASinh {
    T operator()(T x) const {
        return std::asinh(x);
    }
};

template<typename T>
struct ACosh {
    T operator()(T x) const {
        return std::acosh(x);
    }
};

template<typename T>
struct ATanh {
    T operator()(T x) const {
        return std::atanh(x);
    }
};


