/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Client.hpp"

#include "DeviceAvailability.hpp"

#include <fcntl.h>
// POSIX signal handling is not part of the C++ <csignal> interface.
#include <signal.h> // NOLINT(modernize-deprecated-headers)
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef __linux__
#include <sys/prctl.h>
#endif

#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

namespace {

// A signal handler may only communicate through sig_atomic_t.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
volatile sig_atomic_t interrupted = 0;

extern "C" void handleSignal(const int signal) { interrupted = signal; }

constexpr std::string_view USAGE =
    "Usage: mqt-core-qdmi-check --device ID [--timeout SECONDS]\n"
    "Check one registered QDMI device (default timeout: 30 seconds).\n"
    "Exit codes: 0 available, 1 failed, 2 invalid arguments, 124 timeout.\n";

[[nodiscard]] auto check(const std::string& id) -> int {
  try {
    const auto device = qdmi::Session::openDevice(id);
    qdmi::detail::checkDeviceAvailability(device, id, "QDMI device ");
    return 0;
  } catch (...) {
    return 1;
  }
}

[[nodiscard]] auto supervise(const std::string& id,
                             const std::chrono::seconds timeout) -> int {
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
    // The launch validator may need to kill an unresponsive checker.
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
    // Provider diagnostics can contain credentials, URLs, or configuration.
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
    // Normal exit includes driver and provider teardown within the deadline.
    std::exit(check(id)); // NOLINT(concurrency-mt-unsafe)
  }

  // Either side can establish the group before the child loads a provider.
  static_cast<void>(setpgid(child, child));
  int result = 1;
  while (interrupted == 0) {
    siginfo_t status{};
    const auto waited = waitid(P_PID, static_cast<id_t>(child), &status,
                               WEXITED | WNOHANG | WNOWAIT);
    if (waited == 0 && status.si_pid == child) {
      result = status.si_code == CLD_EXITED && status.si_status == 0 ? 0 : 1;
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

} // namespace

int main(const int argc, char** argv) try {
  const auto arguments = std::span(argv, static_cast<size_t>(argc));
  if (argc == 2 && std::string_view(arguments[1]) == "--help") {
    std::cout << USAGE;
    return 0;
  }
  std::string id;
  int seconds = 30;
  bool hasTimeout = false;
  for (size_t i = 1; i < arguments.size(); ++i) {
    const std::string_view option(arguments[i]);
    if (++i == arguments.size()) {
      std::cerr << USAGE;
      return 2;
    }
    const std::string_view value(arguments[i]);
    if (option == "--device" && id.empty() && !value.empty()) {
      id = value;
    } else if (option == "--timeout" && !hasTimeout) {
      const auto parsed =
          std::from_chars(value.data(), value.data() + value.size(), seconds);
      if (parsed.ec != std::errc{} ||
          parsed.ptr != value.data() + value.size() || seconds < 1 ||
          seconds > 3600) {
        std::cerr << USAGE;
        return 2;
      }
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
  const auto result = supervise(id, std::chrono::seconds(seconds));
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
