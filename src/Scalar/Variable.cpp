/**
 * @file src/Scalar/Variable.cpp
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

#include "Expression.hpp"

Variable Variable::t1{(Type)1};
Variable Variable::t0{(Type)0};

// Dummy variables are nameless variables counted negatively
Variable::Variable() : m_nidx{this->m_idx_count++} {
  // Reserve a buffer of expressions
  m_gh_vec.reserve(g_vec_init);
}

// Variables with concrete values
Variable::Variable(const Type &value)
    : m_nidx{this->m_idx_count++}, m_value_var{value} {
  // Set all the values for VarWrap
  m_var.setConstructor(value);
  // Reserve a buffer of expressions
  m_gh_vec.reserve(g_vec_init);
  // Clear cache
  m_cache.clear();
}

// Copy assignment to Type values
Variable &Variable::operator=(const Type &value) {
  // Set value
  m_value_var = value;
  // Set all the values for VarWrap
  m_var.setConstructor(value);
  // A number doesn't contain any content, so clear expression buffer
  m_gh_vec.clear();
  // If the variable is nameless, then just set expression to the value
  if ("0" == m_var.getVariableName()) {
    m_var.setString(ToString(value));
  }
  // Clear cache
  m_cache.clear();
  return *this;
}

// Variable copy constructor
Variable::Variable(const Variable &exp)
    : m_nidx{exp.m_nidx},
      m_cache{exp.m_cache},
      m_var{exp.m_var},
      m_value_var{exp.m_value_var},
      m_gh_vec{exp.m_gh_vec} {
  // Copy visited flag
  m_visited = exp.m_visited;
}

// Variable copy constructor
Variable::Variable(Variable &&exp) noexcept
    : m_nidx{std::exchange(exp.m_nidx, -1)},
      m_cache{std::move(exp.m_cache)},
      m_var{std::move(exp.m_var)},
      m_value_var{std::exchange(exp.m_value_var, (Type)0)},
      m_gh_vec{std::move(exp.m_gh_vec)} {
  // Copy visited flag
  m_visited = std::exchange(exp.m_visited, false);
}

// Copy assignment from one variable to another
Variable &Variable::operator=(const Variable &exp) {
  if (&exp != this) {
    // Copy all members
    m_nidx = exp.m_nidx;
    m_cache = exp.m_cache;
    m_var = exp.m_var;
    m_value_var = exp.m_value_var;
    m_visited = exp.m_visited;
    m_gh_vec = exp.m_gh_vec;
  }
  return *this;
}

Variable &Variable::operator=(Variable &&exp) noexcept {
  m_nidx = std::exchange(exp.m_nidx, -1);
  m_cache = std::move(exp.m_cache);
  m_var = std::move(exp.m_var);
  m_value_var = std::exchange(exp.m_value_var, (Type)0);
  m_gh_vec = std::move(exp.m_gh_vec);
  m_visited = std::exchange(exp.m_visited, false);

  return *this;
}

void Variable::setValue(Type val) { m_var.setValue(val); }

Type Variable::getValue() const { return m_var.getValue(); }

void Variable::setdValue(Type val) { m_var.setdValue(val); }

Type Variable::getdValue() const { return m_var.getdValue(); }

void Variable::setExpression(const std::string &str) {
  m_var.setExpression(str);
}

const std::string &Variable::getExpression() const {
  return m_var.getExpression();
}

void Variable::resetImpl() {
  this->m_visited = true;
  // Reset states
  for (auto &i : m_gh_vec) {
    if (i != nullptr) {
      i->reset();
    }
  }
  this->m_visited = false;
}

/*
* ======================================================================================================
* ======================================================================================================
* ======================================================================================================
 _   _ ___________ _____ _   _  ___   _       _____  _   _ ___________ _ _____
___ ______  _____ | | | |_   _| ___ \_   _| | | |/ _ \ | |     |  _  || | | |
___| ___ \ |   |  _  |/ _ \|  _  \/  ___| | | | | | | | |_/ / | | | | | / /_\ \|
|     | | | || | | | |__ | |_/ / |   | | | / /_\ \ | | |\ `--.
| | | | | | |    /  | | | | | |  _  || |     | | | || | | |  __||    /| |   | |
| |  _  | | | | `--. \ \ \_/ /_| |_| |\ \  | | | |_| | | | || |____ \ \_/ /\ \_/
/ |___| |\ \| |___\ \_/ / | | | |/ / /\__/ /
 \___/ \___/\_| \_| \_/  \___/\_| |_/\_____/  \___/  \___/\____/\_|
\_\_____/\___/\_| |_/___/  \____/

*======================================================================================================
*======================================================================================================
*======================================================================================================
*/

// Evaluate value in run-time
Type Variable::eval() {
  /* eval BEGIN */
  if (false == this->m_visited) {
    // Set value
    setValue(m_value_var);
    // Set visit flag to true
    this->m_visited = true;
    // Loop on internal equations
    for (auto &i : m_gh_vec) {
      if (nullptr != i) {
        setValue(i->eval());
      }
    }
    // Return result
    return getValue();
  }
  /* eval END */

  // Return result
  return getValue();
}

// Evaluate 1st derivative in forward mode
Type Variable::devalF(const Variable &var) {
  /* devalF BEGIN */
  if (false == this->m_visited) {
    // Set visit flag to true
    this->m_visited = true;
    // Loop on internal equations
    for (auto &i : m_gh_vec) {
      if (nullptr != i) {
        setdValue(i->devalF(var));
        setValue(i->eval());
      }
    }
  }
  /* devalF END */

  // Return result
  if (m_nidx == var.m_nidx) {
    return (Type)1;
  } else {
    return getdValue();
  }
}

// Deval in run-time for reverse derivative 1st
Type Variable::devalR(const Variable &var) { return m_cache[var.m_nidx]; }

// Evaluate variable
Variable *Variable::symEval() {
  return ((nullptr != this->mp_tmp) ? (this->mp_tmp) : this);
}

// Forward derivative of variable in forward mode
Variable *Variable::symDeval(const Variable &var) {
  // Set differentiation result to a new variable
  if (auto it = this->mp_dtmp.find(m_nidx); it == this->mp_dtmp.end()) {
    auto tmp = Allocate<Variable>((Type)0);
    this->mp_dtmp[m_nidx] = tmp.get();
  }

  if (false == this->m_visited) {
    // Set visit flag to true
    this->m_visited = true;
    // Loop on internal equations
    for (auto &i : m_gh_vec) {
      if (nullptr != i) {
        mp_dtmp[m_nidx] = i->symDeval(var);
        mp_tmp = i->symEval();
      }
    }
    // Set visit flag to false
    this->m_visited = false;
  }

  // Check for seed value
  if (m_nidx == var.m_nidx) {
    return &t1;
  } else {
    return mp_dtmp[m_nidx];
  }
}

// Exposed to user to compute symbolic differentiation
Expression Variable::SymDiff(const Variable &v) {
  resetImpl();
  return *symDeval(v);
}

// Traverse tree
void Variable::traverse(OMPair *cache) {
  if (false == this->m_visited) {
    this->m_visited = true;
    for (auto &i : m_gh_vec) {
      if (nullptr != i) {
        // Traverse the tree
        i->traverse();
        // Set value
        setValue(i->eval());
        // Save cache
        m_cache = std::move(i->getCache());
      }
    }
  }
}

// Get cache
OMPair &Variable::getCache() { return m_cache; }

// Reset
void Variable::reset() {
  if (true == this->m_visited) {
    this->m_visited = false;
    for (auto &i : m_gh_vec) {
      if (nullptr != i) {
        i->reset();
      }
    }
  }
  // Reset flag
  this->m_visited = false;

  // Set derivative to zero
  setValue(m_value_var);
  setdValue((Type)0);

  // Empty cache
  if (false == m_cache.empty()) {
    m_cache.clear();
  }

  // Reset mp_tmp and mp_dtmp
  MetaVariable::resetTemp();
}

// Get type
std::string_view Variable::getType() const { return "Variable"; }

// Find me
bool Variable::findMe(void *v) const {
  if (static_cast<const void *>(this) == v) {
    return true;
  } else {
    return false;
  }
}

// Destructor
Variable::~Variable() = default;