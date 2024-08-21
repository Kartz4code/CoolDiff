/**
 * @file include/MetaVariable.hpp
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

#include "CommonHeader.hpp"

class MetaVariable {
 protected:
  // Value and derivative temporaries
  Variable *mp_tmp{nullptr};
  UnOrderedMap<size_t, Variable *> mp_dtmp;

  // Reset temporaries
  void resetTemp();

  // Index counter (A counter to count the number of operations)
  inline static size_t m_idx_count{0};

 public:
  // Visited flag
  bool m_visited{false};

  // Default constructor
  MetaVariable() = default;

  /* Type implementations */
  // Evaluate run-time
  V_PURE(Type eval());

  // Forward AD (1st and 2nd derivatives)
  V_PURE(Type devalF(const Variable &));

  // Reverse AD (1st and 2nd derivatives)
  V_PURE(void traverse(OMPair * = nullptr));
  V_PURE(OMPair &getCache());

  /* Variable implementations */
  // Evaluate pseudo-symbolic expression
  V_PURE(Variable *symEval());
  // Differentiate pseudo-symbolic expression
  V_PURE(Variable *symDeval(const Variable &));

  // Reset all visited flags
  V_PURE(void reset());

  // Find me
  V_PURE(bool findMe(void *) const);

  // Get type
  V_PURE(std::string_view getType() const);

  // Destructor
  V_DTR(~MetaVariable()) = default;
};