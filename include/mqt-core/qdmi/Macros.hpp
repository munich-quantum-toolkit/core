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

// Apply a macro to an argument
#define APPLY(func, arg) func(arg)
// Apply a macro to two arguments
#define APPLY2(func, arg1, arg2) func(arg1, arg2)
// Apply a macro to an argument that is already wrapped in parentheses
#define APPLY_PAREN(func, arg) func arg

// Concatenate two tokens
#define CAT(a, b) CAT_I(a, b)
#define CAT_I(a, b) a##b

// TO_SEQ expands a number to a sequence of parentheses of that length
#define TO_SEQ(n) TO_SEQ_I(n)
#define TO_SEQ_I(n) TO_SEQ_##n
#define TO_SEQ_0
#define TO_SEQ_1 ()
#define TO_SEQ_2 () TO_SEQ_1
#define TO_SEQ_3 () TO_SEQ_2
#define TO_SEQ_4 () TO_SEQ_3
#define TO_SEQ_5 () TO_SEQ_4
#define TO_SEQ_6 () TO_SEQ_5
#define TO_SEQ_7 () TO_SEQ_6
#define TO_SEQ_8 () TO_SEQ_7
#define TO_SEQ_9 () TO_SEQ_8

// With HEAD one can extract the first element of a sequence
#define HEAD(x) HEAD_I(HEAD_III x)
#define HEAD_I(x) HEAD_II(x)
#define HEAD_II(x, _) x
#define HEAD_III(x) x, NIL

// With TAIL one can extract the tail of a sequence
#define TAIL(seq) TAIL_I seq
#define TAIL_I(_)

// With EMPTY one can check if a sequence is empty and returns 1 if it is empty
// and 0 otherwise
#define EMPTY(seq) HEAD(CAT(EMPTY_, EMPTY_I seq))
#define EMPTY_I(_) NOT_EMPTY_I
// NOLINTBEGIN(cppcoreguidelines-macro-to-enum)
#define EMPTY_NOT_EMPTY_I (0)
#define EMPTY_EMPTY_I (1)
// NOLINTEND(cppcoreguidelines-macro-to-enum)

#define SIZE(seq) CAT(SIZE_, SIZE_0 seq)
#define SIZE_0(_) SIZE_1
#define SIZE_1(_) SIZE_2
#define SIZE_2(_) SIZE_3
#define SIZE_3(_) SIZE_4
#define SIZE_4(_) SIZE_5
#define SIZE_5(_) SIZE_6
#define SIZE_6(_) SIZE_7
#define SIZE_7(_) SIZE_8
#define SIZE_8(_) SIZE_9

// NOLINTBEGIN(cppcoreguidelines-macro-to-enum)
#define SIZE_SIZE_0 0
#define SIZE_SIZE_1 1
#define SIZE_SIZE_2 2
#define SIZE_SIZE_3 3
#define SIZE_SIZE_4 4
#define SIZE_SIZE_5 5
#define SIZE_SIZE_6 6
#define SIZE_SIZE_7 7
#define SIZE_SIZE_8 8
#define SIZE_SIZE_9 9
// NOLINTEND(cppcoreguidelines-macro-to-enum)

// NTH extracts the i-th element from a sequence
#define NTH(i, seq) NTH_I(NTH_##i seq)
#define NTH_I(e) NTH_II(e)
#define NTH_II(e, _) e
#define NTH_0(e) e, NIL
#define NTH_1(_) NTH_0
#define NTH_2(_) NTH_1
#define NTH_3(_) NTH_2
#define NTH_4(_) NTH_3
#define NTH_5(_) NTH_4
#define NTH_6(_) NTH_5
#define NTH_7(_) NTH_6
#define NTH_8(_) NTH_7

// SUB calculates a - b
#define SUB(a, b) SUB_I(TO_SEQ(a), b)
#define SUB_I(a, b) SIZE(CAT(SUB_, SUB_##b a))
#define SUB_1(_) SUB_0
#define SUB_2(_) SUB_1
#define SUB_3(_) SUB_2
#define SUB_4(_) SUB_3
#define SUB_5(_) SUB_4
#define SUB_6(_) SUB_5
#define SUB_7(_) SUB_6
#define SUB_8(_) SUB_7
#define SUB_9(_) SUB_8
#define SUB_SUB_0
#define SUB_SUB_1
#define SUB_SUB_2
#define SUB_SUB_3
#define SUB_SUB_4
#define SUB_SUB_5
#define SUB_SUB_6
#define SUB_SUB_7
#define SUB_SUB_8
#define SUB_SUB_9

// NOT negates a boolean value where 0 is false and 1 is true
#define NOT(x) CAT(NOT_, x)
// NOLINTBEGIN(cppcoreguidelines-macro-to-enum)
#define NOT_0 1
#define NOT_1 0
// NOLINTEND(cppcoreguidelines-macro-to-enum)
#define BOOL(x) NOT(EMPTY(TO_SEQ(x)))

// ITE is a macro that implements an if-then-else construct
#define ITE(bit, t, f) ITE_I(bit, t, f)
#define ITE_I(bit, t, f) CAT(ITE_, bit(t, f))
#define ITE_0(t, f) f
#define ITE_1(t, f) t

// MIN returns the minimum of two values
#define MIN(a, b) ITE(BOOL(SUB(b, a)), a, b)

// NTH_MAX extracts the i-th element from a sequence, but ensures that i is not
// larger than the size of the sequence minus one. If i is larger than the size
// of the sequence minus one, it returns the last element of the sequence.
#define NTH_MAX(i, seq) NTH_MAX_I(i, SUB(SIZE(seq), 1), seq)
#define NTH_MAX_I(i, n, seq) NTH_MAX_II(MIN(i, n), seq)
#define NTH_MAX_II(i, seq) NTH(i, seq)

// Iterate over a sequence with a macro accepting one element of the sequence at
// a time. The macro is applied to each element of the sequence.
#define ITERATE(macro, seq) CONTINUE_1(macro, (seq))

#define ITERATE_1(macro, seq) ITE(EMPTY(seq), BREAK, CONTINUE_2)(macro, (seq))
#define ITERATE_2(macro, seq) ITE(EMPTY(seq), BREAK, CONTINUE_3)(macro, (seq))
#define ITERATE_3(macro, seq) ITE(EMPTY(seq), BREAK, CONTINUE_4)(macro, (seq))
#define ITERATE_4(macro, seq) ITE(EMPTY(seq), BREAK, CONTINUE_5)(macro, (seq))
#define ITERATE_5(macro, seq) ITE(EMPTY(seq), BREAK, CONTINUE_6)(macro, (seq))
#define ITERATE_6(macro, seq) ITE(EMPTY(seq), BREAK, CONTINUE_7)(macro, (seq))
#define ITERATE_7(macro, seq) ITE(EMPTY(seq), BREAK, CONTINUE_8)(macro, (seq))
#define ITERATE_8(macro, seq) ITE(EMPTY(seq), BREAK, CONTINUE_9)(macro, (seq))
#define ITERATE_9(macro, seq) ITE(EMPTY(seq), BREAK, CONTINUE_10)(macro, (seq))

#define CONTINUE_1(macro, seq)                                                 \
  APPLY(macro, APPLY_PAREN(HEAD, seq)) ITERATE_1(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_2(macro, seq)                                                 \
  APPLY(macro, APPLY_PAREN(HEAD, seq)) ITERATE_2(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_3(macro, seq)                                                 \
  APPLY(macro, APPLY_PAREN(HEAD, seq)) ITERATE_3(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_4(macro, seq)                                                 \
  APPLY(macro, APPLY_PAREN(HEAD, seq)) ITERATE_4(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_5(macro, seq)                                                 \
  APPLY(macro, APPLY_PAREN(HEAD, seq)) ITERATE_5(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_6(macro, seq)                                                 \
  APPLY(macro, APPLY_PAREN(HEAD, seq)) ITERATE_6(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_7(macro, seq)                                                 \
  APPLY(macro, APPLY_PAREN(HEAD, seq)) ITERATE_7(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_8(macro, seq)                                                 \
  APPLY(macro, APPLY_PAREN(HEAD, seq)) ITERATE_8(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_9(macro, seq)                                                 \
  APPLY(macro, APPLY_PAREN(HEAD, seq)) ITERATE_9(macro, APPLY_PAREN(TAIL, seq))

#define BREAK(macro, seq)

// Iterate over a sequence with a macro accepting the index and one element of
// the sequence at a time. The macro is applied to each element of the sequence
// with the index starting at 0.
#define ITERATE_I(macro, seq) CONTINUE_I_1(macro, (seq))

#define ITERATE_I_1(macro, seq)                                                \
  ITE(EMPTY(seq), BREAK, CONTINUE_I_2)(macro, (seq))
#define ITERATE_I_2(macro, seq)                                                \
  ITE(EMPTY(seq), BREAK, CONTINUE_I_3)(macro, (seq))
#define ITERATE_I_3(macro, seq)                                                \
  ITE(EMPTY(seq), BREAK, CONTINUE_I_4)(macro, (seq))
#define ITERATE_I_4(macro, seq)                                                \
  ITE(EMPTY(seq), BREAK, CONTINUE_I_5)(macro, (seq))
#define ITERATE_I_5(macro, seq)                                                \
  ITE(EMPTY(seq), BREAK, CONTINUE_I_6)(macro, (seq))
#define ITERATE_I_6(macro, seq)                                                \
  ITE(EMPTY(seq), BREAK, CONTINUE_I_7)(macro, (seq))
#define ITERATE_I_7(macro, seq)                                                \
  ITE(EMPTY(seq), BREAK, CONTINUE_I_8)(macro, (seq))
#define ITERATE_I_8(macro, seq)                                                \
  ITE(EMPTY(seq), BREAK, CONTINUE_I_9)(macro, (seq))
#define ITERATE_I_9(macro, seq)                                                \
  ITE(EMPTY(seq), BREAK, CONTINUE_I_10)(macro, (seq))

#define CONTINUE_I_1(macro, seq)                                               \
  APPLY2(macro, 0, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_1(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_2(macro, seq)                                               \
  APPLY2(macro, 1, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_2(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_3(macro, seq)                                               \
  APPLY2(macro, 2, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_3(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_4(macro, seq)                                               \
  APPLY2(macro, 3, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_4(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_5(macro, seq)                                               \
  APPLY2(macro, 4, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_5(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_6(macro, seq)                                               \
  APPLY2(macro, 5, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_6(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_7(macro, seq)                                               \
  APPLY2(macro, 6, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_7(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_8(macro, seq)                                               \
  APPLY2(macro, 7, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_8(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_9(macro, seq)                                               \
  APPLY2(macro, 8, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_9(macro, APPLY_PAREN(TAIL, seq))

// Include the device headers for the devices in the list. The device headers
// are expected to be available under `<prefix>_qdmi/device.h`.
// clang-format off
#define INCLUDE(prefix) APPLY(STR, CAT(prefix, _qdmi/device.h))
// clang-format on
#define STR(x) #x

// Casts type `QDMI_<type>` to the prefixed type of the device
// `<prefix>_QDMI_<type>`.
#define PREFIX_CAST(prefix, type, var)                                         \
  /* NOLINTNEXTLINE(bugprone-casting-through-void) */                          \
  static_cast<prefix##_QDMI_##type>(static_cast<void*>(var))
// Same as `PREFIX_CAST`, but for pointers.
#define PREFIX_PTR_CAST(prefix, type, var) PREFIX_CAST(prefix, type*, var)
// Same as `PREFIX_CAST`, but for constant pointers.
// clang-format off
#define PREFIX_CONST_PTR_CAST(prefix, type, var)                               \
/* NOLINTBEGIN(bugprone-casting-through-void,bugprone-macro-parentheses) */  \
static_cast<const prefix## _QDMI_## type*>(static_cast<const void*>(var))    \
/* NOLINTEND(bugprone-casting-through-void,bugprone-macro-parentheses) */
// clang-format on

// NOLINTBEGIN(bugprone-macro-parentheses)

// Within a query function add the corresponding conditional branches to return
// the requested property value.
// Args:
//   prop_name: is the QDMI enum value of the property to query.
//   prop_type: is the type of the property value, e.g., `int`.
//   prop_value: is the value of the property to return.
//   prop: is the property parameter of the query function.
//   size: is the size of the value buffer.
//   value: is the pointer to the value buffer to write the property value to.
//   size_ret: is the pointer to the size of the value buffer to write the size
//     of the property value to.
#define ADD_SINGLE_VALUE_PROPERTY(prop_name, prop_type, prop_value, prop,      \
                                  size, value, size_ret)                       \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < sizeof(prop_type)) {                                      \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        *static_cast<prop_type*>(value) = prop_value;                          \
      }                                                                        \
      if ((size_ret) != nullptr) {                                             \
        *size_ret = sizeof(prop_type);                                         \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  }

// Same as `ADD_SINGLE_VALUE_PROPERTY`, but for properties that are strings.
// Note:
//   The `prop_value` must be a c-type string. Otherwise, the purpose of the
//   parameters can be retrieved from the `ADD_SINGLE_VALUE_PROPERTY`macro.
#define ADD_STRING_PROPERTY(prop_name, prop_value, prop, size, value,          \
                            size_ret)                                          \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < strlen(prop_value) + 1) {                                 \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        strncpy(static_cast<char*>(value), prop_value, size);                  \
        /* NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) */  \
        static_cast<char*>(value)[size - 1] = '\0';                            \
      }                                                                        \
      if ((size_ret) != nullptr) {                                             \
        *size_ret = strlen(prop_value) + 1;                                    \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  }

// Same as `ADD_SINGLE_VALUE_PROPERTY`, but for properties that are lists of
// values.
// Note:
//   The `prop_values` must be a `std::vector<prop_type>` or a similar container
//   providing the functions `size()` and `data()`. Otherwise, the purpose of
//   the parameters can be retrieved from the `ADD_SINGLE_VALUE_PROPERTY` macro.
#define ADD_LIST_PROPERTY(prop_name, prop_type, prop_values, prop, size,       \
                          value, size_ret)                                     \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < (prop_values).size() * sizeof(prop_type)) {               \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        memcpy(static_cast<void*>(value),                                      \
               static_cast<const void*>((prop_values).data()),                 \
               (prop_values).size() * sizeof(prop_type));                      \
      }                                                                        \
      if ((size_ret) != nullptr) {                                             \
        *size_ret = (prop_values).size() * sizeof(prop_type);                  \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  }

// Within a function to set a parameter, add the corresponding conditional
// branches to set the parameter accordingly.
// Args:
//   param_name: is the QDMI enum value of the parameter to set.
//   param_type: is the type of the parameter value, e.g., `int`.
//   var: is the variable to set the parameter value to.
//   param: is the parameter argument of the setter function.
//   size: is the size of the value buffer.
//   value: is the pointer to the value buffer to read the parameter value from.
#define ADD_SINGLE_VALUE_PARAMETER(param_name, param_type, var, param, size,   \
                                   value)                                      \
  {                                                                            \
    if ((param) == (param_name)) {                                             \
      if ((value) != nullptr) {                                                \
        if ((size) < sizeof(param_type)) {                                     \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        var = *static_cast<const param_type*>(value);                          \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  }

// Same as `ADD_SINGLE_VALUE_PARAMETER`, but for parameters that are held in a
// pointer.
// Note:
//   The `var` must be a pointer to the type of the parameter value and must be
//   allocated before calling the setter function. Otherwise, the purpose of the
//   parameters can be retrieved from the `ADD_SINGLE_VALUE_PARAMETER` macro.
#define ADD_POINTER_PARAMETER(param_name, ptr_type, var, param, size, value)   \
  {                                                                            \
    if ((param) == (param_name)) {                                             \
      if ((value) != nullptr) {                                                \
        if ((size) == 0) {                                                     \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        memcpy(static_cast<void*>(var), static_cast<const void*>(value),       \
               size);                                                          \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  }
// NOLINTEND(bugprone-macro-parentheses)
