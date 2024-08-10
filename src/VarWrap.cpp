#include "VarWrap.hpp"

VarWrap::VarWrap(Type val) : m_var_name{ "x" + std::to_string(m_count++) }, 
                             m_expression{ m_var_name }, 
                             m_value{ val }, 
                             m_dvalue{ (Type)0 }
{}

VarWrap::VarWrap() : m_var_name{ "0" }, 
                     m_expression{ "0" },
                     m_value{ (Type)0 }, 
                     m_dvalue{ (Type)0 }
{}


// Copy constructor 
VarWrap::VarWrap(const VarWrap& vw) : m_expression{ vw.m_expression },
                                      m_value{ vw.m_value },
                                      m_dvalue{ vw.m_dvalue },
                                      m_var_name{ vw.m_var_name } 
{}

// Move constructor 
VarWrap::VarWrap(VarWrap&& vw) noexcept : m_expression{ std::move(vw.m_expression) },
                                          m_value{ std::exchange(vw.m_value, {}) },
                                          m_dvalue{ std::exchange(vw.m_dvalue, {}) },
                                          m_var_name { std::move(vw.m_var_name) } 
{}

// Copy assignment 
VarWrap& VarWrap::operator=(const VarWrap& vw) {
    m_expression = vw.m_expression;
    m_value = vw.m_value;
    m_dvalue = vw.m_dvalue;
    m_var_name = vw.m_var_name;
    return *this;
}

// Move assignment
VarWrap& VarWrap::operator=(VarWrap&& vw) noexcept {
    m_expression = std::move(vw.m_expression);
    m_value = std::exchange(vw.m_value, {});
    m_dvalue = std::exchange(vw.m_dvalue, {});
    m_var_name = std::move(vw.m_var_name);
    return *this;
}

// Set constructor
void VarWrap::setConstructor(Type val) {
    m_var_name = { "x" + std::to_string(m_count++) };
    m_expression = { m_var_name };
    m_value = { val };
    m_dvalue = { (Type)0 };
}

const std::string& VarWrap::getVariableName() const {
    return m_var_name;
}

void VarWrap::setExpression(const std::string& str) {
    m_expression = str;
}

const std::string& VarWrap::getExpression() const {
    return m_expression;
}

const Type VarWrap::getValue() const {
    return m_value;
}

void VarWrap::setValue(Type val) {
    m_value = val;
}

const Type VarWrap::getdValue() const {
    return m_dvalue;
}

void VarWrap::setdValue(Type val) {
    m_dvalue = val;
}

void VarWrap::setString(const std::string& str) {
    m_expression = str;
}