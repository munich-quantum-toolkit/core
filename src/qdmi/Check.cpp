/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/QDMI.hpp"
#include "qdmi/common/Common.hpp"

#include "qdmi/constants.h"

#ifdef _WIN32
#include <memory>
#include <windows.h>
#else
#include <fcntl.h>
// POSIX signal handling is not part of the C++ <csignal> interface.
#include <signal.h> // NOLINT(modernize-deprecated-headers)
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifdef __linux__
#include <linux/prctl.h>
#include <sys/prctl.h>
#endif

#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

namespace {

// A signal handler may only communicate through sig_atomic_t.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
volatile std::sig_atomic_t interrupted = 0;

extern "C" void handleSignal(const int signal) { interrupted = signal; }

constexpr std::string_view USAGE =
    "Usage: mqt-core-qdmi-check --device ID [--timeout SECONDS] "
    "[--manifest PATH]...\n"
    "Probe whether a registered QDMI device is operational (default timeout: "
    "30 seconds).\n"
    "Exit codes: 0 available, 1 failed, 2 invalid arguments, 124 timeout.\n";

[[nodiscard]] auto check(const std::string& id,
                         const std::span<const std::filesystem::path> manifests)
    -> int {
  try {
    for (const auto& manifest : manifests) {
      try {
        qdmi::builtin_driver::addManifest(manifest);
      } catch (...) {
        // Match Python discovery's best-effort handling of installed manifests.
        continue;
      }
    }
    const auto device = qdmi::Session::openDevice(id);
    const auto status = device.getStatus();
    return status == QDMI_DEVICE_STATUS_IDLE ||
                   status == QDMI_DEVICE_STATUS_BUSY
               ? 0
               : 1;
  } catch (...) {
    return 1;
  }
}

#ifdef _WIN32
[[nodiscard]] auto supervise(const std::chrono::seconds timeout) -> int {
  if (std::signal(SIGTERM, handleSignal) == SIG_ERR ||
      std::signal(SIGINT, handleSignal) == SIG_ERR) {
    return 1;
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  using Handle = std::unique_ptr<void, decltype(&CloseHandle)>;
  const Handle job(CreateJobObjectW(nullptr, nullptr), CloseHandle);
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!job ||
      !SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation,
                               &limits, sizeof(limits))) {
    return 1;
  }

  std::wstring executable(MAX_PATH, L'\0');
  while (true) {
    const auto length = GetModuleFileNameW(
        nullptr, executable.data(), static_cast<DWORD>(executable.size()));
    if (length == 0) {
      return 1;
    }
    if (length < executable.size()) {
      executable.resize(length);
      break;
    }
    executable.resize(executable.size() * 2);
  }
  // Preserve the original argument quoting after replacing argv[0].
  std::wstring_view arguments(GetCommandLineW());
  bool quoted = false;
  while (!arguments.empty()) {
    const auto character = arguments.front();
    if (!quoted && (character == L' ' || character == L'\t')) {
      break;
    }
    if (character == L'"') {
      quoted = !quoted;
    }
    arguments.remove_prefix(1);
  }
  auto command = L"\"" + executable + L"\" --worker" + std::wstring(arguments);
  SECURITY_ATTRIBUTES security{};
  security.nLength = sizeof(security);
  security.bInheritHandle = TRUE;
  const auto sink = CreateFileW(L"NUL", GENERIC_READ | GENERIC_WRITE,
                                FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
                                OPEN_EXISTING, 0, nullptr);
  if (sink == INVALID_HANDLE_VALUE) {
    return 1;
  }
  const Handle output(sink, CloseHandle);
  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = output.get();
  startup.StartupInfo.hStdOutput = output.get();
  startup.StartupInfo.hStdError = output.get();
  SIZE_T attributeBytes = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &attributeBytes);
  std::vector<std::byte> attributes(attributeBytes);
  startup.lpAttributeList =
      reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
  if (!InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0,
                                         &attributeBytes)) {
    return 1;
  }
  // Assign the job during creation so supervisor death cannot orphan a worker.
  auto jobHandle = job.get();
  PROCESS_INFORMATION information{};
  const auto created =
      UpdateProcThreadAttribute(startup.lpAttributeList, 0,
                                PROC_THREAD_ATTRIBUTE_JOB_LIST, &jobHandle,
                                sizeof(jobHandle), nullptr, nullptr) &&
      CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
                     EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW, nullptr,
                     nullptr, &startup.StartupInfo, &information);
  DeleteProcThreadAttributeList(startup.lpAttributeList);
  if (!created) {
    return 1;
  }
  const Handle process(information.hProcess, CloseHandle);
  const Handle thread(information.hThread, CloseHandle);
  int result = 1;
  while (interrupted == 0) {
    const auto waited = WaitForSingleObject(process.get(), 10);
    if (waited == WAIT_OBJECT_0) {
      DWORD status = 1;
      return GetExitCodeProcess(process.get(), &status) && status == 0 ? 0 : 1;
    }
    if (waited == WAIT_FAILED) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      result = 124;
      break;
    }
  }
  TerminateJobObject(job.get(), 1);
  WaitForSingleObject(process.get(), INFINITE);
  return result;
}
#else
[[nodiscard]] auto
supervise(const std::string& id, const std::chrono::seconds timeout,
          const std::span<const std::filesystem::path> manifests) -> int {
  struct sigaction action{};
  action.sa_handler = handleSignal;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGTERM, &action, nullptr) != 0 ||
      sigaction(SIGINT, &action, nullptr) != 0) {
    return 1;
  }

  const auto deadline = std::chrono::steady_clock::now() + timeout;
#ifdef __linux__
  const auto supervisor = getpid();
#endif
  const auto child = fork();
  if (child < 0) {
    return 1;
  }
  if (child == 0) {
#ifdef __linux__
    // Stop the worker if its supervisor is killed.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) != 0 || getppid() != supervisor) {
      _exit(1);
    }
#endif
    action.sa_handler = SIG_DFL;
    const rlimit coreLimit{.rlim_cur = 0, .rlim_max = 0};
    if (setpgid(0, 0) != 0 || sigaction(SIGTERM, &action, nullptr) != 0 ||
        sigaction(SIGINT, &action, nullptr) != 0 ||
        setrlimit(RLIMIT_CORE, &coreLimit) != 0) {
      _exit(1);
    }
    // Device diagnostics can contain credentials, URLs, or configuration.
    // POSIX open takes no mode argument when it does not create a file.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    const auto sink = open("/dev/null", O_RDWR);
    if (sink < 0 || dup2(sink, STDIN_FILENO) < 0 ||
        dup2(sink, STDOUT_FILENO) < 0 || dup2(sink, STDERR_FILENO) < 0) {
      _exit(1);
    }
    if (sink > STDERR_FILENO) {
      close(sink);
    }
    // Run process-exit handlers within the worker deadline.
    std::exit(check(id, manifests)); // NOLINT(concurrency-mt-unsafe)
  }

  // Either side can establish the group before the child loads a device.
  static_cast<void>(setpgid(child, child));
  int result = 1;
  while (interrupted == 0) {
    // clang-tidy 23 does not map glibc's wait definitions to <sys/wait.h>.
    // NOLINTBEGIN(misc-include-cleaner)
    siginfo_t status{};
    const auto waited = waitid(P_PID, static_cast<id_t>(child), &status,
                               WEXITED | WNOHANG | WNOWAIT);
    if (waited == 0 && status.si_pid == child) {
      result = status.si_code == CLD_EXITED && status.si_status == 0 ? 0 : 1;
      // NOLINTEND(misc-include-cleaner)
      break;
    }
    if (waited < 0 && errno != EINTR) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      result = 124;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  // Keep the worker unreaped until group cleanup so its PID cannot be reused.
  static_cast<void>(kill(-child, SIGKILL));
  static_cast<void>(kill(child, SIGKILL));
  while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {
  }
  return result;
}
#endif

} // namespace

#ifdef _WIN32
int wmain(const int argc, wchar_t** argv) try {
  std::vector<std::string> utf8Arguments;
  for (const auto* argument : std::span(argv, static_cast<size_t>(argc))) {
    utf8Arguments.emplace_back(qdmi::detail::pathToString(argument));
  }
  const auto arguments = std::span(utf8Arguments);
  bool worker = false;
#else
int main(const int argc, char** argv) try {
  const auto arguments = std::span(argv, static_cast<size_t>(argc));
#endif
  if (argc == 2 && std::string_view(arguments[1]) == "--help") {
    std::cout << USAGE;
    return 0;
  }
  std::string id;
  std::vector<std::filesystem::path> manifests;
  int seconds = 30;
  bool hasTimeout = false;
  for (size_t i = 1; i < arguments.size(); ++i) {
    const std::string_view option(arguments[i]);
#ifdef _WIN32
    if (option == "--worker" && !worker) {
      worker = true;
      continue;
    }
#endif
    if (++i == arguments.size()) {
      std::cerr << USAGE;
      return 2;
    }
    const std::string_view value(arguments[i]);
    if (option == "--device" && id.empty() && !value.empty()) {
      id = value;
    } else if (option == "--manifest" && !value.empty()) {
      manifests.emplace_back(qdmi::detail::pathFromString(value));
    } else if (option == "--timeout" && !hasTimeout) {
      // from_chars requires a pointer range within the string view.
      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      const auto parsed =
          std::from_chars(value.data(), value.data() + value.size(), seconds);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != value.data() + value.size() || seconds < 1 ||
          seconds > 3600) {
        std::cerr << USAGE;
        return 2;
      }
      // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      hasTimeout = true;
    } else {
      std::cerr << USAGE;
      return 2;
    }
  }
  if (id.empty()) {
    std::cerr << USAGE;
    return 2;
  }
#ifdef _WIN32
  if (worker) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    return check(id, manifests);
  }
  const auto result = supervise(std::chrono::seconds(seconds));
#else
  const auto result = supervise(id, std::chrono::seconds(seconds), manifests);
#endif
  if (result == 124) {
    std::cerr << "QDMI device check timed out\n";
  } else if (result != 0) {
    std::cerr << "QDMI device check failed\n";
  }
  return result;
} catch (...) {
  std::cerr << "QDMI device check failed\n";
  return 1;
}
