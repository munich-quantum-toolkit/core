/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief An example driver implementation in C++.
 */

#define DEVICE_LIST_UPPERCASE (MQT_NA)
#define DEVICE_LIST_LOWERCASE (mqt_na)

#include "qdmi/Driver.hpp"

#include "qdmi/client.h"
#include "qdmi/device.h"

#include <cstddef>
#include <dlfcn.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define APPLY(func, arg) func(arg)
#define APPLY2(func, arg1, arg2) func(arg1, arg2)
#define APPLY_PAREN(func, arg) func arg

#define CAT(a, b) CAT_I(a, b)
#define CAT_I(a, b) a##b

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

#define HEAD(x) HEAD_I(HEAD_III x)
#define HEAD_I(x) HEAD_II(x)
#define HEAD_II(x, _) x
#define HEAD_III(x) x, NIL

#define TAIL(seq) TAIL_I seq
#define TAIL_I(_)

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

#define NOT(x) CAT(NOT_, x)
// NOLINTBEGIN(cppcoreguidelines-macro-to-enum)
#define NOT_0 1
#define NOT_1 0
// NOLINTEND(cppcoreguidelines-macro-to-enum)
#define BOOL(x) NOT(EMPTY(TO_SEQ(x)))

#define ITE(bit, t, f) ITE_I(bit, t, f)
#define ITE_I(bit, t, f) CAT(ITE_, bit(t, f))
#define ITE_0(t, f) f
#define ITE_1(t, f) t

#define MIN(a, b) ITE(BOOL(SUB(b, a)), a, b)

#define NTH_MAX(i, seq) NTH_MAX_I(i, SUB(SIZE(seq), 1), seq)
#define NTH_MAX_I(i, n, seq) NTH_MAX_II(MIN(i, n), seq)
#define NTH_MAX_II(i, seq) NTH(i, seq)

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
  APPLY2(macro, 1, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_1(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_2(macro, seq)                                               \
  APPLY2(macro, 2, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_2(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_3(macro, seq)                                               \
  APPLY2(macro, 3, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_3(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_4(macro, seq)                                               \
  APPLY2(macro, 4, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_4(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_5(macro, seq)                                               \
  APPLY2(macro, 5, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_5(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_6(macro, seq)                                               \
  APPLY2(macro, 6, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_6(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_7(macro, seq)                                               \
  APPLY2(macro, 7, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_7(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_8(macro, seq)                                               \
  APPLY2(macro, 8, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_8(macro, APPLY_PAREN(TAIL, seq))
#define CONTINUE_I_9(macro, seq)                                               \
  APPLY2(macro, 9, APPLY_PAREN(HEAD, seq))                                     \
  ITERATE_I_9(macro, APPLY_PAREN(TAIL, seq))

// clang-format off
#define INCLUDE(prefix) APPLY(STR, CAT(prefix, _qdmi/device.h))
// clang-format on
#define STR(x) #x

// NOLINTBEGIN(readability-duplicate-include)
#include INCLUDE(NTH_MAX(0, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(1, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(2, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(3, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(4, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(5, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(6, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(7, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(8, DEVICE_LIST_LOWERCASE))
#include INCLUDE(NTH_MAX(9, DEVICE_LIST_LOWERCASE))
// NOLINTEND(readability-duplicate-include)

#define PREFIX_CAST(prefix, type, var)                                         \
  /* NOLINTNEXTLINE(bugprone-casting-through-void) */                          \
  static_cast<prefix##_QDMI_##type>(static_cast<void*>(var))
#define PREFIX_PTR_CAST(prefix, type, var) PREFIX_CAST(prefix, type*, var)
// clang-format off
#define PREFIX_CONST_PTR_CAST(prefix, type, var)                               \
  /* NOLINTBEGIN(bugprone-casting-through-void,bugprone-macro-parentheses) */  \
  static_cast<const prefix## _QDMI_## type*>(static_cast<const void*>(var))    \
  /* NOLINTEND(bugprone-casting-through-void,bugprone-macro-parentheses) */
// clang-format on

#define ADD_DEVICE(prefix)                                                     \
  int prefix##_device_initialize(void) {                                       \
    return prefix##_QDMI_device_initialize();                                  \
  }                                                                            \
  int prefix##_device_finalize(void) {                                         \
    return prefix##_QDMI_device_finalize();                                    \
  }                                                                            \
  int prefix##_device_session_alloc(QDMI_Device_Session* session) {            \
    return prefix##_QDMI_device_session_alloc(                                 \
        PREFIX_PTR_CAST(prefix, Device_Session, session));                     \
  }                                                                            \
  int prefix##_device_session_set_parameter(                                   \
      QDMI_Device_Session session, QDMI_Device_Session_Parameter param,        \
      size_t size, const void* value) {                                        \
    return prefix##_QDMI_device_session_set_parameter(                         \
        PREFIX_CAST(prefix, Device_Session, session), param, size, value);     \
  }                                                                            \
  int prefix##_device_session_init(QDMI_Device_Session session) {              \
    return prefix##_QDMI_device_session_init(                                  \
        PREFIX_CAST(prefix, Device_Session, session));                         \
  }                                                                            \
  void prefix##_device_session_free(QDMI_Device_Session session) {             \
    return prefix##_QDMI_device_session_free(                                  \
        PREFIX_CAST(prefix, Device_Session, session));                         \
  }                                                                            \
  int prefix##_device_session_query_device_property(                           \
      QDMI_Device_Session session, QDMI_Device_Property prop, size_t size,     \
      void* value, size_t* size_ret) {                                         \
    return prefix##_QDMI_device_session_query_device_property(                 \
        PREFIX_CAST(prefix, Device_Session, session), prop, size, value,       \
        size_ret);                                                             \
  }                                                                            \
  int prefix##_device_session_query_site_property(                             \
      QDMI_Device_Session session, QDMI_Site site, QDMI_Site_Property prop,    \
      size_t size, void* value, size_t* size_ret) {                            \
    return prefix##_QDMI_device_session_query_site_property(                   \
        PREFIX_CAST(prefix, Device_Session, session),                          \
        PREFIX_CAST(prefix, Site, site), prop, size, value, size_ret);         \
  }                                                                            \
  int prefix##_device_session_query_operation_property(                        \
      QDMI_Device_Session session, QDMI_Operation operation, size_t num_sites, \
      const QDMI_Site* sites, size_t num_params, const double* params,         \
      QDMI_Operation_Property prop, size_t size, void* value,                  \
      size_t* size_ret) {                                                      \
    return prefix##_QDMI_device_session_query_operation_property(              \
        PREFIX_CAST(prefix, Device_Session, session),                          \
        PREFIX_CAST(prefix, Operation, operation), num_sites,                  \
        PREFIX_CONST_PTR_CAST(prefix, Site, sites), num_params, params, prop,  \
        size, value, size_ret);                                                \
  }                                                                            \
  int prefix##_device_session_create_device_job(QDMI_Device_Session session,   \
                                                QDMI_Device_Job* job) {        \
    return prefix##_QDMI_device_session_create_device_job(                     \
        PREFIX_CAST(prefix, Device_Session, session),                          \
        PREFIX_PTR_CAST(prefix, Device_Job, job));                             \
  }                                                                            \
  int prefix##_device_job_set_parameter(QDMI_Device_Job job,                   \
                                        QDMI_Device_Job_Parameter param,       \
                                        size_t size, const void* value) {      \
    return prefix##_QDMI_device_job_set_parameter(                             \
        PREFIX_CAST(prefix, Device_Job, job), param, size, value);             \
  }                                                                            \
  int prefix##_device_job_submit(QDMI_Device_Job job) {                        \
    return prefix##_QDMI_device_job_submit(                                    \
        PREFIX_CAST(prefix, Device_Job, job));                                 \
  }                                                                            \
  int prefix##_device_job_cancel(QDMI_Device_Job job) {                        \
    return prefix##_QDMI_device_job_cancel(                                    \
        PREFIX_CAST(prefix, Device_Job, job));                                 \
  }                                                                            \
  int prefix##_device_job_check(QDMI_Device_Job job,                           \
                                QDMI_Job_Status* status) {                     \
    return prefix##_QDMI_device_job_check(                                     \
        PREFIX_CAST(prefix, Device_Job, job), status);                         \
  }                                                                            \
  int prefix##_device_job_wait(QDMI_Device_Job job) {                          \
    return prefix##_QDMI_device_job_wait(                                      \
        PREFIX_CAST(prefix, Device_Job, job));                                 \
  }                                                                            \
  int prefix##_device_job_get_results(QDMI_Device_Job job,                     \
                                      QDMI_Job_Result result, size_t size,     \
                                      void* data, size_t* size_ret) {          \
    return prefix##_QDMI_device_job_get_results(                               \
        PREFIX_CAST(prefix, Device_Job, job), result, size, data, size_ret);   \
  }                                                                            \
  void prefix##_device_job_free(QDMI_Device_Job job) {                         \
    return prefix##_QDMI_device_job_free(                                      \
        PREFIX_CAST(prefix, Device_Job, job));                                 \
  }

namespace {
ITERATE(ADD_DEVICE, DEVICE_LIST_UPPERCASE)

enum class SessionStatus : uint8_t {
  ALLOCATED,  ///< The session has been allocated but not initialized
  INITIALIZED ///< The session has been initialized and is ready for use
};

/**
 * @brief Definition of the QDMI Library.
 */
struct DeviceLibrary {
  void* libHandle = nullptr;

  // NOLINTBEGIN(readability-identifier-naming)
  /// Function pointer to @ref QDMI_device_initialize.
  decltype(QDMI_device_initialize)* device_initialize{};
  /// Function pointer to @ref QDMI_device_finalize.
  decltype(QDMI_device_finalize)* device_finalize{};
  /// Function pointer to @ref QDMI_device_session_alloc.
  decltype(QDMI_device_session_alloc)* device_session_alloc{};
  /// Function pointer to @ref QDMI_device_session_init.
  decltype(QDMI_device_session_init)* device_session_init{};
  /// Function pointer to @ref QDMI_device_session_free.
  decltype(QDMI_device_session_free)* device_session_free{};
  /// Function pointer to @ref QDMI_device_session_set_parameter.
  decltype(QDMI_device_session_set_parameter)* device_session_set_parameter{};
  /// Function pointer to @ref QDMI_device_session_create_device_job.
  decltype(QDMI_device_session_create_device_job)*
      device_session_create_device_job{};
  /// Function pointer to @ref QDMI_device_job_free.
  decltype(QDMI_device_job_free)* device_job_free{};
  /// Function pointer to @ref QDMI_device_job_set_parameter.
  decltype(QDMI_device_job_set_parameter)* device_job_set_parameter{};
  /// Function pointer to @ref QDMI_device_job_submit.
  decltype(QDMI_device_job_submit)* device_job_submit{};
  /// Function pointer to @ref QDMI_device_job_cancel.
  decltype(QDMI_device_job_cancel)* device_job_cancel{};
  /// Function pointer to @ref QDMI_device_job_check.
  decltype(QDMI_device_job_check)* device_job_check{};
  /// Function pointer to @ref QDMI_device_job_wait.
  decltype(QDMI_device_job_wait)* device_job_wait{};
  /// Function pointer to @ref QDMI_device_job_get_results.
  decltype(QDMI_device_job_get_results)* device_job_get_results{};
  /// Function pointer to @ref QDMI_device_session_query_device_property.
  decltype(QDMI_device_session_query_device_property)*
      device_session_query_device_property{};
  /// Function pointer to @ref QDMI_device_session_query_site_property.
  decltype(QDMI_device_session_query_site_property)*
      device_session_query_site_property{};
  /// Function pointer to @ref QDMI_device_session_query_operation_property.
  decltype(QDMI_device_session_query_operation_property)*
      device_session_query_operation_property{};
  // NOLINTEND(readability-identifier-naming)

  // default constructor
  DeviceLibrary() = default;

  // delete copy constructor, copy assignment, move constructor, move assignment
  // to allow only one instance and proper destruction of the dynamic library.
  DeviceLibrary(const DeviceLibrary&) = delete;
  DeviceLibrary& operator=(const DeviceLibrary&) = delete;
  DeviceLibrary(DeviceLibrary&&) = delete;
  DeviceLibrary& operator=(DeviceLibrary&&) = delete;

  // destructor
  ~DeviceLibrary() {
    // Check if QDMI_device_finalize is not NULL before calling it.
    if (device_finalize != nullptr) {
      device_finalize();
    }
    // close the dynamic library
    if (libHandle != nullptr) {
      dlclose(libHandle);
    }
  }
};
} // namespace

/**
 * @brief Definition of the QDMI Device.
 */
struct QDMI_Device_impl_d {
  const DeviceLibrary* library = nullptr;
  QDMI_Device_Session session = nullptr;

  // destructor
  ~QDMI_Device_impl_d() {
    if (library != nullptr && session != nullptr) {
      library->device_session_free(session);
    }
  }
};

/**
 * @brief Definition of the QDMI Session.
 */
struct QDMI_Session_impl_d {
  SessionStatus status = SessionStatus::ALLOCATED;
};

/**
 * @brief Definition of the QDMI Job.
 */
struct QDMI_Job_impl_d {
  QDMI_Device_Job deviceJob = nullptr;
  QDMI_Device device = nullptr;
};

namespace {

[[nodiscard]] auto staticDeviceLibraries()
    -> std::array<std::unique_ptr<DeviceLibrary>,
                  SIZE(DEVICE_LIST_UPPERCASE)>& {
  static std::array<std::unique_ptr<DeviceLibrary>, SIZE(DEVICE_LIST_UPPERCASE)>
      libraries;
  return libraries;
}

[[nodiscard]] auto dynamicDeviceLibraries()
    -> std::unordered_map<void*, std::unique_ptr<DeviceLibrary>>& {
  static std::unordered_map<void*, std::unique_ptr<DeviceLibrary>> libraries;
  return libraries;
}

#define ADD_STATIC_DEVICE_LIBRARY(id, prefix)                                  \
  auto& library =                                                              \
      *(staticDeviceLibraries()[id] = std::make_unique<DeviceLibrary>());      \
  /* static library has no handle */                                           \
  library.libHandle = nullptr;                                                 \
  /* load the function symbols from the dynamic library */                     \
  library.device_initialize = prefix##_##device_initialize;                    \
  library.device_finalize = prefix##_##device_finalize;                        \
  /* device session interface */                                               \
  library.device_session_alloc = prefix##_##device_session_alloc;              \
  library.device_session_init = prefix##_##device_session_init;                \
  library.device_session_free = prefix##_##device_session_free;                \
  library.device_session_set_parameter =                                       \
      prefix##_##device_session_set_parameter;                                 \
  /* device job interface */                                                   \
  library.device_session_create_device_job =                                   \
      prefix##_##device_session_create_device_job;                             \
  library.device_job_free = prefix##_##device_job_free;                        \
  library.device_job_set_parameter = prefix##_##device_job_set_parameter;      \
  library.device_job_submit = prefix##_##device_job_submit;                    \
  library.device_job_cancel = prefix##_##device_job_cancel;                    \
  library.device_job_check = prefix##_##device_job_check;                      \
  library.device_job_wait = prefix##_##device_job_wait;                        \
  library.device_job_get_results = prefix##_##device_job_get_results;          \
  /* device query interface */                                                 \
  library.device_session_query_device_property =                               \
      prefix##_##device_session_query_device_property;                         \
  library.device_session_query_site_property =                                 \
      prefix##_##device_session_query_site_property;                           \
  library.device_session_query_operation_property =                            \
      prefix##_##device_session_query_operation_property;                      \
  /* initialize the device */                                                  \
  library.device_initialize();

#define LOAD_SYMBOL(library, prefix, symbol)                                   \
  {                                                                            \
    const std::string symbolName = std::string(prefix) + "_QDMI_" + #symbol;   \
    (library).symbol = reinterpret_cast<decltype((library).symbol)>(           \
        dlsym((library).libHandle, symbolName.c_str()));                       \
    if ((library).symbol == nullptr) {                                         \
      throw std::runtime_error("Failed to load symbol: " + symbolName);        \
    }                                                                          \
  }

void addDynamicDeviceLibrary(const std::string& libName,
                             const std::string& prefix) {
  auto* libHandle = dlopen(libName.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (libHandle == nullptr) {
    throw std::runtime_error("Couldn't open the device library: " + libName);
  }
  if (const auto it = dynamicDeviceLibraries().find(libHandle);
      it != dynamicDeviceLibraries().end()) {
    // dlopen uses reference counting, so we need to decrement the reference
    // count that was increased by dlopen.
    dlclose(libHandle);
    return;
  }
  auto it = dynamicDeviceLibraries()
                .emplace(libHandle, std::make_unique<DeviceLibrary>())
                .first;
  auto& library = *it->second;
  library.libHandle = libHandle;

  try {
    // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
    // load the function symbols from the dynamic library
    LOAD_SYMBOL(library, prefix, device_initialize)
    LOAD_SYMBOL(library, prefix, device_finalize)
    // device session interface
    LOAD_SYMBOL(library, prefix, device_session_alloc)
    LOAD_SYMBOL(library, prefix, device_session_init)
    LOAD_SYMBOL(library, prefix, device_session_free)
    LOAD_SYMBOL(library, prefix, device_session_set_parameter)
    // device job interface
    LOAD_SYMBOL(library, prefix, device_session_create_device_job)
    LOAD_SYMBOL(library, prefix, device_job_free)
    LOAD_SYMBOL(library, prefix, device_job_set_parameter)
    LOAD_SYMBOL(library, prefix, device_job_submit)
    LOAD_SYMBOL(library, prefix, device_job_cancel)
    LOAD_SYMBOL(library, prefix, device_job_check)
    LOAD_SYMBOL(library, prefix, device_job_wait)
    LOAD_SYMBOL(library, prefix, device_job_get_results)
    // device query interface
    LOAD_SYMBOL(library, prefix, device_session_query_device_property)
    LOAD_SYMBOL(library, prefix, device_session_query_site_property)
    LOAD_SYMBOL(library, prefix, device_session_query_operation_property)
    // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
  } catch (const std::exception&) {
    dlclose(libHandle);
    throw;
  }
  // initialize the device
  library.device_initialize();
}

[[nodiscard]] auto devices()
    -> std::vector<std::unique_ptr<QDMI_Device_impl_d>>& {
  static std::vector<std::unique_ptr<QDMI_Device_impl_d>> devices;
  return devices;
}

auto addDevice(const DeviceLibrary& library) -> void {
  auto& device =
      *devices().emplace_back(std::make_unique<QDMI_Device_impl_d>());
  device.library = &library;
  device.library->device_session_alloc(&device.session);
  device.library->device_session_init(device.session);
}

[[nodiscard]] auto sessions()
    -> std::unordered_map<QDMI_Session, std::unique_ptr<QDMI_Session_impl_d>>& {
  static std::unordered_map<QDMI_Session, std::unique_ptr<QDMI_Session_impl_d>>
      sessions;
  return sessions;
}

} // namespace

namespace na {
auto initialize(const std::vector<Library>& additionalLibraries) -> void {
  // Initialize known static device libraries
  ITERATE_I(ADD_STATIC_DEVICE_LIBRARY, DEVICE_LIST_UPPERCASE)
  // Load additional dynamic device libraries
  for (const auto& [prefix, path] : additionalLibraries) {
    addDynamicDeviceLibrary(path, prefix);
  }
  // Add all static and dynamic device libraries to the device list
  for (const auto& lib : staticDeviceLibraries()) {
    addDevice(*lib);
  }
  for (const auto& [handle, lib] : dynamicDeviceLibraries()) {
    addDevice(*lib);
  }
}

auto finalize() -> void {
  while (!sessions().empty()) {
    QDMI_session_free(sessions().begin()->first);
  }
  devices().clear();
  dynamicDeviceLibraries().clear();
  staticDeviceLibraries().fill(nullptr);
}
} // namespace na

int QDMI_session_alloc(QDMI_Session* session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  auto uniqueSession = std::make_unique<QDMI_Session_impl_d>();
  *session = sessions()
                 .emplace(uniqueSession.get(), std::move(uniqueSession))
                 .first->first;
  return QDMI_SUCCESS;
}

int QDMI_session_init(QDMI_Session session) {
  if (session == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  session->status = SessionStatus::INITIALIZED;
  return QDMI_SUCCESS;
}

void QDMI_session_free(QDMI_Session session) { sessions().erase(session); }

int QDMI_session_set_parameter(QDMI_Session session,
                               QDMI_Session_Parameter param, const size_t size,
                               const void* value) {
  if (session == nullptr || (value != nullptr && size == 0) ||
      param >= QDMI_SESSION_PARAMETER_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::ALLOCATED) {
    return QDMI_ERROR_BADSTATE;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_session_query_session_property(QDMI_Session session,
                                        QDMI_Session_Property prop, size_t size,
                                        void* value, size_t* sizeRet) {
  if (session == nullptr || (value != nullptr && size == 0) ||
      prop >= QDMI_SESSION_PROPERTY_MAX) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  if (session->status != SessionStatus::INITIALIZED) {
    return QDMI_ERROR_BADSTATE;
  }
  if (prop == (QDMI_SESSION_PROPERTY_DEVICES)) {
    if (value != nullptr) {
      if (size < devices().size() * sizeof(QDMI_Device)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      memcpy(value, static_cast<const void*>(devices().data()),
             devices().size() * sizeof(QDMI_Device));
    }
    if (sizeRet != nullptr) {
      *sizeRet = devices().size() * sizeof(QDMI_Device);
    }
    return QDMI_SUCCESS;
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

int QDMI_device_create_job(QDMI_Device dev, QDMI_Job* job) {
  if (dev == nullptr || job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  *job = new QDMI_Job_impl_d();
  (*job)->device = dev;
  return dev->library->device_session_create_device_job(dev->session,
                                                        &(*job)->deviceJob);
}

void QDMI_job_free(QDMI_Job job) {
  if (job != nullptr) {
    job->device->library->device_job_free(job->deviceJob);
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    delete job;
  }
}

int QDMI_job_set_parameter(QDMI_Job job, QDMI_Job_Parameter param,
                           const size_t size, const void* value) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->library->device_job_set_parameter(
      job->deviceJob, static_cast<QDMI_Device_Job_Parameter>(param), size,
      value);
}

int QDMI_job_submit(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->library->device_job_submit(job->deviceJob);
}

int QDMI_job_cancel(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->library->device_job_cancel(job->deviceJob);
}

int QDMI_job_check(QDMI_Job job, QDMI_Job_Status* status) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->library->device_job_check(job->deviceJob, status);
}

int QDMI_job_wait(QDMI_Job job) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->library->device_job_wait(job->deviceJob);
}

int QDMI_job_get_results(QDMI_Job job, QDMI_Job_Result result,
                         const size_t size, void* data, size_t* sizeRet) {
  if (job == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return job->device->library->device_job_get_results(job->deviceJob, result,
                                                      size, data, sizeRet);
}

int QDMI_device_query_device_property(QDMI_Device device,
                                      QDMI_Device_Property prop,
                                      const size_t size, void* value,
                                      size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->library->device_session_query_device_property(
      device->session, prop, size, value, sizeRet);
}

int QDMI_device_query_site_property(QDMI_Device device, QDMI_Site site,
                                    QDMI_Site_Property prop, const size_t size,
                                    void* value, size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->library->device_session_query_site_property(
      device->session, site, prop, size, value, sizeRet);
}

int QDMI_device_query_operation_property(
    QDMI_Device device, QDMI_Operation operation, const size_t numSites,
    const QDMI_Site* sites, const size_t numParams, const double* params,
    QDMI_Operation_Property prop, const size_t size, void* value,
    size_t* sizeRet) {
  if (device == nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  return device->library->device_session_query_operation_property(
      device->session, operation, numSites, sites, numParams, params, prop,
      size, value, sizeRet);
}
