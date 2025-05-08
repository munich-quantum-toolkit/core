/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "ir/Definitions.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace sym {
static constexpr double TOLERANCE = 1e-9;

class SymbolicException final : public std::invalid_argument {
  std::string msg;

public:
  explicit SymbolicException(std::string m)
      : std::invalid_argument("Symbolic Exception"), msg(std::move(m)) {}

  [[nodiscard]] const char* what() const noexcept override {
    return msg.c_str();
  }
};

/**
 * @brief Struct representing a symbolic variable.
 * @details Variable names are assigned ids during creation. These ids are
 * statically stored in a map. The name of a variable can be retrieved by its
 * id. Variables are lexicographically ordered by their ids.
 */
struct Variable {
  // NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
  static inline std::unordered_map<std::string, std::size_t> registered{};
  static inline std::unordered_map<std::size_t, std::string> names{};
  static inline std::size_t nextId{};
  // NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

  /**
   * @brief Construct variable with given name.
   * @param name Name of the variable.
   */
  explicit Variable(const std::string& name);

  /**
   * @brief Get the name of the variable.
   * @return Name of the variable.
   */
  [[nodiscard]] std::string getName() const noexcept;

  /**
   * @brief Check whether this variable's id is equal to another variable's id.
   * @param rhs Variable to compare with.
   * @return True if the variables are equal, false otherwise.
   */
  bool operator==(const Variable& rhs) const { return id == rhs.id; }

  /**
   * @brief Check whether this variable's id is not equal to another variable's
   * id.
   * @param rhs Variable to compare with.
   * @return True if the variables are not equal, false otherwise.
   */
  bool operator!=(const Variable& rhs) const { return !((*this) == rhs); }

  /**
   * @brief Check whether this variable's id is less than another variable's id
   * with respect to the default lexicographic ordering.
   * @param rhs Variable to compare with.
   * @return True if this variable's id is less than the other variable's id,
   * false otherwise.
   */
  bool operator<(const Variable& rhs) const { return id < rhs.id; }

  /**
   * @brief Check whether this variable's id is greater than another variable's
   * id with respect to the default lexicographic ordering.
   * @param rhs Variable to compare with.
   * @return True if this variable's id is greater than the other variable's id,
   * false otherwise.
   */
  bool operator>(const Variable& rhs) const { return id > rhs.id; }

private:
  std::size_t id{};
};
} // namespace sym

/**
 * @brief Hash function for the Variable struct.
 */
template <> struct std::hash<sym::Variable> {
  std::size_t operator()(const sym::Variable& var) const noexcept {
    return std::hash<std::string>()(var.getName());
  }
};

namespace sym {

/**
 * @brief Type alias for a variable assignment. Maps variables to their assigned
 * values.
 */
using VariableAssignment = std::unordered_map<Variable, double>;

/**
 * @brief Struct representing a symbolic term. A term is a variable multiplied
 * by a coefficient.
 * @tparam T Type of the coefficient. Must be constructible from an integer and
 * a double.
 */
template <typename T,
          typename = std::enable_if_t<std::is_constructible_v<int, T> &&
                                      std::is_constructible_v<T, double>>>
class Term {
public:
  /**
   * @brief Get the variable of the term.
   * @return Variable of the term.
   */
  [[nodiscard]] Variable getVar() const noexcept { return var; }

  /**
   * @brief Get the coefficient of the term.
   * @return Coefficient of the term.
   */
  [[nodiscard]] T getCoeff() const noexcept { return coeff; }

  /**
   * @brief Check whether the term has a zero coefficient.
   * @return True if the coefficient is zero, false otherwise.
   */
  [[nodiscard]] bool hasZeroCoeff() const {
    return std::abs(static_cast<double>(coeff)) < TOLERANCE;
  }

  /**
   * @brief Construct a term with a given variable and coefficient.
   * @param v Variable of the term.
   * @param coef Coefficient of the term.
   */
  explicit Term(const Variable v, T coef = 1.) : coeff(coef), var(v) {};

  /**
   * @brief Get the negative of the term.
   * @return Negative of the term.
   */
  Term operator-() const { return Term(var, -coeff); }

  /**
   * @brief Add to the coefficient of the term.
   * @param rhs Value to add to the coefficient.
   */
  void addCoeff(const T& rhs) { coeff += rhs; }

  /**
   * @brief Multiply the term by a scalar (add coefficients).
   * @param rhs Scalar to multiply the term by.
   * @return Reference to the term.
   */
  Term& operator*=(const T& rhs) {
    coeff *= rhs;
    return *this;
  }

  /**
   * @brief Divide the term by a scalar (divide coefficients).
   * @param rhs Scalar to divide the term by.
   * @return Reference to the term.
   */
  Term& operator/=(const T& rhs) {
    coeff /= rhs;
    return *this;
  }

  /**
   * @brief Multiply the term by a scalar (add coefficients).
   * @param rhs Scalar to multiply the term by.
   * @return Reference to the term.
   */
  Term& operator/=(const std::int64_t rhs) {
    coeff /= static_cast<T>(rhs);
    return *this;
  }

  /**
   * @brief Check whether the Term's variable is assigned in a given assignment.
   * @param assignment Assignment to check.
   * @return True if the variable is assigned, false otherwise.
   */
  [[nodiscard]] bool
  totalAssignment(const VariableAssignment& assignment) const {
    return assignment.find(getVar()) != assignment.end();
  }

  /**
   * @brief Evaluate the term with respect to a given assignment.
   * @param assignment Assignment to evaluate the term with.
   * @return Result of the evaluation.
   */
  [[nodiscard]] double evaluate(const VariableAssignment& assignment) const {
    if (!totalAssignment(assignment)) {
      throw SymbolicException("Cannot instantiate variable " +
                              getVar().getName() + ". No value given.");
    }
    return assignment.at(getVar()) * getCoeff();
  }

private:
  T coeff;
  Variable var;
};

/** @name Term operators
 *  @brief Overloaded operators for the Term struct.
 * @param lhs Left-hand side of the operator.
 * @param rhs Right-hand side of the operator.
 * @return Term obtained by applying the operation.
 */
///@{
template <typename T,
          typename = std::enable_if_t<std::is_constructible_v<int, T>>>
Term<T> operator*(Term<T> lhs, const double rhs) {
  lhs *= rhs;
  return lhs;
}
template <typename T,
          typename = std::enable_if_t<std::is_constructible_v<int, T>>>
Term<T> operator/(Term<T> lhs, const double rhs) {
  lhs /= rhs;
  return lhs;
}
template <typename T,
          typename = std::enable_if_t<std::is_constructible_v<int, T>>>
Term<T> operator*(double lhs, const Term<T>& rhs) {
  return rhs * lhs;
}
template <typename T,
          typename = std::enable_if_t<std::is_constructible_v<int, T>>>
Term<T> operator/(double lhs, const Term<T>& rhs) {
  return rhs / lhs;
}
///@}

/**
 * @brief Check whether two terms are equal.
 * @details Two terms are equal if their variables are equal and their
 * coefficients are equal within a small tolerance.
 * @param lhs Left-hand side of the operator.
 * @param rhs Right-hand side of the operator.
 * @return True if the terms are equal, false otherwise.
 */
template <typename T> bool operator==(const Term<T>& lhs, const Term<T>& rhs) {
  return lhs.getVar() == rhs.getVar() &&
         std::abs(lhs.getCoeff() - rhs.getCoeff()) < TOLERANCE;
}

/**
 * @brief Check whether two terms are not equal.
 * @details Two terms are not equal if their variables are not equal or their
 * coefficients are not equal within a small tolerance.
 * @param lhs Left-hand side of the operator.
 * @param rhs Right-hand side of the operator.
 * @return True if the terms are not equal, false otherwise.
 */
template <typename T> bool operator!=(const Term<T>& lhs, const Term<T>& rhs) {
  return !(lhs == rhs);
}
} // namespace sym

/**
 * @brief Hash function for the Term struct.
 */
template <typename T> struct std::hash<sym::Term<T>> {
  std::size_t operator()(const sym::Term<T>& term) const noexcept {
    const auto h1 = std::hash<sym::Variable>{}(term.getVar());
    const auto h2 = std::hash<T>{}(term.getCoeff());
    return qc::combineHash(h1, h2);
  }
};

namespace sym {

/**
 * @brief Class representing a symbolic expression. An expression is a sum of
 * terms and a constant.
 * @tparam T Type of the coefficients of the terms. Must be constructible from
 * an integer and a double.
 * @tparam U Type of the constant. Must be constructible from a double.
 */
template <
    typename T, typename U,
    typename = std::enable_if_t<
        std::is_constructible_v<T, U> && std::is_constructible_v<U, T> &&
        std::is_constructible_v<int, T> && std::is_constructible_v<T, double> &&
        std::is_constructible_v<U, double>>>
class Expression {
public:
  using iterator = typename std::vector<Term<T>>::iterator;
  using const_iterator = typename std::vector<Term<T>>::const_iterator;

  /**
   * @brief Construct an Expression from a varargs list of terms. Constant is
   * set to 0.
   * @tparam Args Variadic template parameter for the terms.
   * @param t First term.
   * @param ms Remaining terms.
   */
  template <typename... Args> explicit Expression(Term<T> t, Args&&... ms) {
    terms.emplace_back(t);
    (terms.emplace_back(std::forward<Args>(ms)), ...);
    sortTerms();
    aggregateEqualTerms();
  }

  /**
   * @brief Construct an Expression from a varargs list of variables. Constant
   * is set to 0 and coefficients of the variables are set to 1.
   * @tparam Args Variadic template parameter for the variables.
   * @param v First variable.
   * @param ms Remaining variables.
   */
  template <typename... Args> explicit Expression(Variable v, Args&&... ms) {
    terms.emplace_back(Term<T>(v));
    (terms.emplace_back(std::forward<Args>(ms)), ...);
    sortTerms();
    aggregateEqualTerms();
  }

  /**
   * @brief Construct an Expression from a vector of terms and a constant.
   * @param ts Vector of terms.
   * @param con Constant of the expression.
   */
  Expression(const std::vector<Term<T>>& ts, const U& con)
      : terms(ts), constant(con) {};

  /**
   * @brief Default constructor. Expression has no Terms and a constant of 0.
   */
  Expression() = default;

  /**
   * @brief Construct an Expression from a constant. Expression has no Terms.
   * @param r Constant of the expression.
   */
  explicit Expression(const U& r) : constant(r) {};

  iterator begin() { return terms.begin(); }
  [[nodiscard]] const_iterator begin() const { return terms.cbegin(); }
  iterator end() { return terms.end(); }
  [[nodiscard]] const_iterator end() const { return terms.cend(); }
  [[nodiscard]] const_iterator cbegin() const { return terms.cbegin(); }
  [[nodiscard]] const_iterator cend() const { return terms.cend(); }

  /**
   * @brief Check whether the expression is zero, i.e. has no Terms and 0
   * constant.
   * @return True if the expression is zero, false otherwise.
   */
  [[nodiscard]] bool isZero() const {
    return terms.empty() && constant == U{T{0}};
  }

  /**
   * @brief Check whether the expression is a constant, i.e. has no Terms.
   * @return True if the expression is a constant, false otherwise.
   */
  [[nodiscard]] bool isConstant() const { return terms.empty(); }

  /**
   * @brief Add two expressions. Coefficients of like terms and constants are
   * added.
   * @details If a term for the same variable is already present in the
   * expression, the coefficients are added. Otherwise, the term is inserted
   * into the expression.
   * @param rhs Expression to add.
   * @return Reference to the expression.
   */
  Expression& operator+=(const Expression& rhs) {
    if (this->isZero()) {
      *this = rhs;
      return *this;
    }

    if (rhs.isZero()) {
      return *this;
    }

    auto t = rhs.begin();

    while (t != rhs.end()) {
      const auto insertPos =
          std::lower_bound(terms.begin(), terms.end(), *t,
                           [&](const Term<T>& lhs, const Term<T>& r) {
                             return lhs.getVar() < r.getVar();
                           });
      if (insertPos != terms.end() && insertPos->getVar() == t->getVar()) {
        if (std::abs(insertPos->getCoeff() + t->getCoeff()) < TOLERANCE) {
          terms.erase(insertPos);
        } else {
          insertPos->addCoeff(t->getCoeff());
        }
      } else {
        terms.insert(insertPos, *t);
      }
      ++t;
    }
    constant += rhs.constant;
    return *this;
  }

  /**
   * @brief Add a term to the expression. Coefficients of like terms are added.
   * @details If a term for the same variable is already present in the
   * expression, the coefficients are added. Otherwise, the term is inserted
   * into the expression.
   * @param rhs Term to add.
   * @return Reference to the expression.
   */
  Expression<T, U>& operator+=(const Term<T>& rhs) {
    return *this += Expression(rhs);
  }

  /**
   * @brief Add a constant to the expression.
   * @param rhs Constant to add.
   * @return Reference to the expression.
   */
  Expression<T, U>& operator+=(const U& rhs) {
    constant += rhs;
    return *this;
  }

  /**
   * @brief Subtract two expressions. Coefficients of like terms and constants
   * are subtracted.
   * @details If a term for the same variable is already present in the
   * expression, the coefficients are subtracted. Otherwise, the term is
   * inserted into the expression.
   * @param rhs Expression to subtract.
   * @return Reference to the expression.
   */
  Expression<T, U>& operator-=(const Expression<T, U>& rhs) {
    return *this += -rhs;
  }

  /**
   * @brief Subtract a term from the expression. Coefficients of like terms are
   * subtracted.
   * @details If a term for the same variable is already present in the
   * expression, the coefficients are subtracted. Otherwise, the term is
   * inserted into the expression.
   * @param rhs Term to subtract.
   * @return Reference to the expression.
   */
  Expression<T, U>& operator-=(const Term<T>& rhs) { return *this += -rhs; }

  /**
   * @brief Subtract a constant from the expression.
   * @param rhs Constant to subtract.
   * @return Reference to the expression.
   */
  Expression<T, U>& operator-=(const U& rhs) { return *this += -rhs; }

  /** @name Multiplication operators
   * @brief Multiply the expression by a scalar. Multiplies the coefficients of
   * the terms and the constant by the scalar.
   * @param rhs Scalar to multiply the expression by.
   * @return Reference to the expression.
   */
  ///@{
  Expression<T, U>& operator*=(const T& rhs) {
    if (std::abs(static_cast<double>(rhs)) < TOLERANCE) {
      terms.clear();
      constant = U{T{0}};
      return *this;
    }
    std::for_each(terms.begin(), terms.end(), [&](auto& term) { term *= rhs; });
    constant = U{double{constant} * double{rhs}};
    return *this;
  }

  template <typename V = U, std::enable_if_t<!std::is_same_v<T, V>, int> = 0>
  Expression<T, U>& operator*=(const U& rhs) {
    if (std::abs(static_cast<double>(T{rhs})) < TOLERANCE) {
      terms.clear();
      constant = U{T{0}};
      return *this;
    }
    std::for_each(terms.begin(), terms.end(),
                  [&](auto& term) { term *= T{rhs}; });
    constant *= rhs;
    return *this;
  }
  ///@}

  /** @name Division operators
   * @brief Divide the expression by a scalar. Divides the coefficients of the
   * terms and the constant by the scalar.
   * @details Throws an exception if the scalar is zero.
   * @param rhs Scalar to divide the expression by.
   * @return Reference to the expression.
   */
  ///@{
  Expression<T, U>& operator/=(const T& rhs) {
    if (std::abs(static_cast<double>(T{rhs})) < TOLERANCE) {
      throw std::runtime_error("Trying to divide expression by 0!");
    }
    std::for_each(terms.begin(), terms.end(), [&](auto& term) { term /= rhs; });
    constant = U{double{constant} / double{rhs}};
    return *this;
  }

  template <typename V = U, std::enable_if_t<!std::is_same_v<T, V>, int> = 0>
  Expression<T, U>& operator/=(const U& rhs) {
    if (std::abs(static_cast<double>(T{rhs})) < TOLERANCE) {
      throw std::runtime_error("Trying to divide expression by 0!");
    }
    std::for_each(terms.begin(), terms.end(),
                  [&](auto& term) { term /= T{rhs}; });
    constant /= rhs;
    return *this;
  }

  Expression<T, U>& operator/=(int64_t rhs) {
    if (rhs == 0) {
      throw std::runtime_error("Trying to divide expression by 0!");
    }
    std::for_each(terms.begin(), terms.end(),
                  [&](auto& term) { term /= T{static_cast<double>(rhs)}; });
    constant = U{double{constant} / static_cast<double>(rhs)};
    return *this;
  }

  ///@}

  /**
   * @brief Get the negative of the expression.
   * @details Negates the coefficients of the terms and the constant.
   * @return Negative of the expression.
   */
  [[nodiscard]] Expression<T, U> operator-() const {
    Expression<T, U> e;
    e.terms.reserve(terms.size());
    for (auto& t : terms) {
      e.terms.push_back(-t);
    }
    e.constant = -constant;
    return e;
  }

  /**
   * @brief Get the term at a given index.
   * @details No bounds checking is performed.
   * @param i Index of the term.
   * @return Term at the given index.
   */
  [[nodiscard]] const Term<T>& operator[](const std::size_t i) const {
    return terms[i];
  }

  /**
   * @brief Get the constant of the expression.
   * @return Constant of the expression.
   */
  [[nodiscard]] U getConst() const noexcept { return constant; }

  /**
   * @brief Set the constant of the expression.
   * @param val Constant to set.
   */
  void setConst(const U& val) { constant = val; }
  [[nodiscard]] auto numTerms() const { return terms.size(); }

  /**
   * @brief Get the terms of the expression.
   * @return Terms of the expression.
   */
  [[nodiscard]] const std::vector<Term<T>>& getTerms() const { return terms; }

  /**
   * @brief Get the variables appearing in terms of the expression.
   * @return Variables of the expression.
   */
  [[nodiscard]] std::unordered_set<Variable> getVariables() const {
    auto vars = std::unordered_set<Variable>{};
    for (const auto& term : terms) {
      vars.insert(term.getVar());
    }
    return vars;
  }

  /**
   * @brief Convert the expression to a different type.
   * @details Converts the coefficients of the terms and the constant to a
   * different type.
   * @tparam V Type to convert to.
   * @return Expression with coefficients and constant converted to type V.
   */
  template <typename V,
            std::enable_if_t<std::is_constructible_v<U, V>>* = nullptr>
  Expression<T, V> convert() const {
    return Expression<T, V>(terms, V{constant});
  }

  /**
   * @brief Evaluate the expression with respect to a given assignment.
   * @param assignment Assignment to evaluate the expression with.
   * @return Result of the evaluation.
   */
  [[nodiscard]] double evaluate(const VariableAssignment& assignment) const {
    auto initial = static_cast<double>(constant);
    return std::accumulate(terms.begin(), terms.end(), initial,
                           [&](const double sum, const auto& term) {
                             return term.evaluate(assignment) + sum;
                           });
  }

private:
  std::vector<Term<T>> terms;
  U constant{T{0.0}};

  void sortTerms() {
    std::sort(terms.begin(), terms.end(),
              [&](const Term<T>& lhs, const Term<T>& rhs) {
                return lhs.getVar() < rhs.getVar();
              });
  }

  void aggregateEqualTerms() {
    for (auto t = terms.begin(); t != terms.end();) {
      auto next = std::next(t);
      while (next != terms.end() && t->getVar() == next->getVar()) {
        t->addCoeff(next->getCoeff());
        next = terms.erase(next);
      }
      if (t->hasZeroCoeff()) {
        t = terms.erase(t);
      } else {
        t = next;
      }
    }
  }
};

/** @name Arithmetic Expression operators
 *  @brief Overloaded operators for the Expression struct.
 * @param lhs Left-hand side of the operator.
 * @param rhs Right-hand side of the operator.
 * @return Expression obtained by applying the operation.
 */
///@{
template <typename T, typename U>
Expression<T, U> operator+(Expression<T, U> lhs, const Expression<T, U>& rhs) {
  lhs += rhs;
  return lhs;
}

template <typename T, typename U>
Expression<T, U> operator+(Expression<T, U> lhs, const Term<T>& rhs) {
  lhs += rhs;
  return lhs;
}

template <typename T, typename U>
Expression<T, U> operator+(const Term<T>& lhs, Expression<T, U> rhs) {
  rhs += lhs;
  return rhs;
}

template <typename T, typename U>
Expression<T, U> operator+(const U& lhs, Expression<T, U> rhs) {
  rhs += lhs;
  return rhs;
}

template <typename T, typename U>
Expression<T, U> operator+(Expression<T, U> lhs, const U& rhs) {
  lhs += rhs;
  return lhs;
}

template <typename T, typename U>
Expression<T, U> operator+([[maybe_unused]] const T& lhs,
                           Expression<T, U> rhs) {
  rhs += rhs;
  return rhs;
}

template <typename T, typename U>
Expression<T, U> operator-(Expression<T, U> lhs, const Expression<T, U>& rhs) {
  lhs -= rhs;
  return lhs;
}
template <typename T, typename U>
Expression<T, U> operator-(Expression<T, U> lhs, const Term<T>& rhs) {
  lhs -= rhs;
  return lhs;
}
template <typename T, typename U>
Expression<T, U> operator-(const Term<T>& lhs, Expression<T, U> rhs) {
  rhs -= lhs;
  return rhs;
}
template <typename T, typename U>
Expression<T, U> operator-(const U& lhs, Expression<T, U> rhs) {
  rhs -= lhs;
  return rhs;
}

template <typename T, typename U>
Expression<T, U> operator-(Expression<T, U> lhs, const U& rhs) {
  lhs -= rhs;
  return lhs;
}

template <typename T, typename U>
Expression<T, U> operator*(Expression<T, U> lhs, const T& rhs) {
  lhs *= rhs;
  return lhs;
}

template <typename T, typename U,
          std::enable_if_t<!std::is_same_v<T, U>>* = nullptr>
Expression<T, U> operator*(Expression<T, U> lhs, const U& rhs) {
  lhs *= rhs;
  return lhs;
}

template <typename T, typename U>
Expression<T, U> operator/(Expression<T, U> lhs, const T& rhs) {
  lhs /= rhs;
  return lhs;
}

template <typename T, typename U,
          std::enable_if_t<!std::is_same_v<T, U>>* = nullptr>
Expression<T, U> operator/(Expression<T, U> lhs, const U& rhs) {
  lhs /= rhs;
  return lhs;
}

template <typename T, typename U>
Expression<T, U> operator/(Expression<T, U> lhs, int64_t rhs) {
  lhs /= rhs;
  return lhs;
}

template <typename T, typename U>
Expression<T, U> operator*(const T& lhs, Expression<T, U> rhs) {
  return rhs * lhs;
}

template <typename T, typename U,
          std::enable_if_t<!std::is_same_v<T, U>>* = nullptr>
Expression<T, U> operator*(const U& lhs, Expression<T, U> rhs) {
  return rhs * lhs;
}

///@}

/**
 * @brief Check whether two expressions are equal.
 * @param lhs Left-hand side of the operator.
 * @param rhs Right-hand side of the operator.
 * @return True if the expressions are equal, false otherwise.
 */
template <typename T, typename U>
bool operator==(const Expression<T, U>& lhs, const Expression<T, U>& rhs) {
  if (lhs.numTerms() != rhs.numTerms() || lhs.getConst() != rhs.getConst()) {
    return false;
  }

  const auto lhsTerms = lhs.numTerms();
  for (size_t i = 0; i < lhsTerms; ++i) {
    if (std::abs(lhs[i].getCoeff() - rhs[i].getCoeff()) >= TOLERANCE) {
      return false;
    }
  }
  return true;
}

/**
 * @brief Check whether two expressions are not equal.
 * @param lhs Left-hand side of the operator.
 * @param rhs Right-hand side of the operator.
 * @return True if the expressions are not equal, false otherwise.
 */
template <typename T, typename U>
bool operator!=(const Expression<T, U>& lhs, const Expression<T, U>& rhs) {
  return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& os, const Variable& var);

template <typename T>
std::ostream& operator<<(std::ostream& os, const Term<T>& term) {
  os << term.getCoeff() << "*" << term.getVar().getName();
  return os;
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const Expression<T, U>& expr) {
  std::for_each(expr.begin(), expr.end(),
                [&](const auto& term) { os << term << " + "; });
  os << expr.getConst();
  return os;
}
} // namespace sym

template <typename T, typename U> struct std::hash<sym::Expression<T, U>> {
  std::size_t operator()(const sym::Expression<T, U>& expr) const noexcept {
    std::size_t seed = 0U;
    for (const auto& term : expr) {
      qc::hashCombine(seed, std::hash<sym::Term<T>>{}(term));
    }
    qc::hashCombine(seed, std::hash<U>{}(expr.getConst()));
    return seed;
  }
}; // namespace std

namespace qc {
using Symbolic = sym::Expression<fp, fp>;
using VariableAssignment = std::unordered_map<sym::Variable, fp>;
using SymbolOrNumber = std::variant<Symbolic, fp>;
} // namespace qc
