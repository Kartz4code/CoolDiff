#include "Expression.hpp"

Expression::Expression() {
    Variable::m_nidx = this->m_idx_count++;
    // Reserve a buffer of expressions
    Variable::m_gh_vec.reserve(g_vec_init);
    // Emplace the expression in a generic holder
    Variable::m_gh_vec.emplace_back(&Variable::t0);
}

Expression::Expression(const Expression& exp) {
    Variable::m_gh_vec.emplace_back(&exp);
}

Expression& Expression::SymDiff(const Variable& var) {
    auto tmp = Allocate<Expression>();
    *tmp = Variable::SymDiff(var);
    return *tmp;
}

// Get type
std::string_view Expression::getType() const {
    return "Expression";
}

Expression::~Expression() = default;
