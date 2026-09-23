/*
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <signal.h> // NOLINT(modernize-deprecated-headers)
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace mqt::spank {

class Validation final {
  static constexpr size_t MAX_ENVIRONMENT_BYTES = 128UL * 1024;
  enum class State : uint8_t { Empty, Copying, Checking, Available, Failed };
  struct Shared {
    std::atomic<State> state{State::Empty};
    size_t size = 0;
    std::array<char, MAX_ENVIRONMENT_BYTES> snapshot{};
  };
  static_assert(std::atomic<State>::is_always_lock_free);

public:
  Validation() = default;
  Validation(const Validation&) = delete;
  Validation& operator=(const Validation&) = delete;
  Validation(Validation&&) = delete;
  Validation& operator=(Validation&&) = delete;

  ~Validation() {
    if (shared_ != nullptr) {
      munmap(shared_, sizeof(Shared));
    }
  }

  void prepare() {
    auto* memory = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (memory == MAP_FAILED) {
      throw std::runtime_error("could not prepare launch validation");
    }
    shared_ = std::construct_at(static_cast<Shared*>(memory));
  }

  void check(const std::string& executable, const int timeout,
             const std::string& device, std::vector<std::string> environment) {
    if (shared_ == nullptr) {
      throw std::runtime_error("launch validation was not prepared");
    }
    std::vector<std::string_view> ordered(environment.begin(),
                                          environment.end());
    std::ranges::sort(ordered);
    std::string snapshot = device + '\0';
    for (const auto entry : ordered) {
      if (!entry.starts_with("SLURM_") && !entry.starts_with("SLURMD_")) {
        snapshot += entry;
        snapshot += '\0';
      }
      if (snapshot.size() > MAX_ENVIRONMENT_BYTES) {
        throw std::runtime_error("validation environment exceeds 128 KiB");
      }
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(timeout + 2);
    auto expected = State::Empty;
    if (shared_->state.compare_exchange_strong(expected, State::Copying)) {
      std::memcpy(shared_->snapshot.data(), snapshot.data(), snapshot.size());
      shared_->size = snapshot.size();
      shared_->state.store(State::Checking, std::memory_order_release);
      const bool available = execute(executable, timeout, device, environment);
      shared_->state.store(available ? State::Available : State::Failed,
                           std::memory_order_release);
    }

    while (true) {
      auto state = shared_->state.load(std::memory_order_acquire);
      if (state >= State::Checking &&
          (shared_->size != snapshot.size() ||
           std::memcmp(shared_->snapshot.data(), snapshot.data(),
                       snapshot.size()) != 0)) {
        state = execute(executable, timeout, device, environment)
                    ? State::Available
                    : State::Failed;
      }
      if (state == State::Available) {
        return;
      }
      if (state == State::Failed) {
        throw std::runtime_error("QDMI launch validation failed");
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        throw std::runtime_error("QDMI launch validation timed out");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

private:
  static bool execute(const std::string& executable, const int timeout,
                      const std::string& device,
                      std::vector<std::string>& environment) {
    const auto seconds = std::to_string(timeout);
    // execve does not modify the strings despite its mutable argument types.
    // NOLINTBEGIN(cppcoreguidelines-pro-type-const-cast)
    std::array arguments{
        const_cast<char*>(executable.c_str()), const_cast<char*>("--device"),
        const_cast<char*>(device.c_str()),     const_cast<char*>("--timeout"),
        const_cast<char*>(seconds.c_str()),    static_cast<char*>(nullptr),
    };
    // NOLINTEND(cppcoreguidelines-pro-type-const-cast)
    std::vector<char*> envp;
    envp.reserve(environment.size() + 1);
    for (auto& entry : environment) {
      envp.push_back(entry.data());
    }
    envp.push_back(nullptr);

    const auto parent = getpid();
    const auto child = fork();
    if (child < 0) {
      return false;
    }
    if (child == 0) {
      // Slurm's caught handlers survive fork until execve resets them.
      struct sigaction action{};
      action.sa_handler = SIG_DFL;
      sigemptyset(&action.sa_mask);
      sigset_t mask{};
      sigemptyset(&mask);
      sigaddset(&mask, SIGTERM);
      if (sigaction(SIGTERM, &action, nullptr) != 0 ||
          sigprocmask(SIG_UNBLOCK, &mask, nullptr) != 0) {
        _exit(1);
      }
      // prctl uses the Linux variadic C ABI.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
      if (prctl(PR_SET_PDEATHSIG, SIGTERM) != 0 || getppid() != parent ||
          setpgid(0, 0) != 0) {
        _exit(1);
      }
      // Never send provider or launcher diagnostics to Slurm's log stream.
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
      const auto sink = open("/dev/null", O_RDWR);
      if (sink < 0 || dup2(sink, STDIN_FILENO) < 0 ||
          dup2(sink, STDOUT_FILENO) < 0 || dup2(sink, STDERR_FILENO) < 0) {
        _exit(1);
      }
      if (sink > STDERR_FILENO) {
        close(sink);
      }
      execve(executable.c_str(), arguments.data(), envp.data());
      _exit(1);
    }
    static_cast<void>(setpgid(child, child));
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(timeout + 1);
    bool available = false;
    while (true) {
      siginfo_t status{};
      const auto result = waitid(P_PID, static_cast<id_t>(child), &status,
                                 WEXITED | WNOHANG | WNOWAIT);
      if (result == 0 && status.si_pid == child) {
        available = status.si_code == CLD_EXITED && status.si_status == 0;
        break;
      }
      if ((result < 0 && errno != EINTR) ||
          std::chrono::steady_clock::now() >= deadline) {
        static_cast<void>(kill(child, SIGTERM));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // Reap only after cleanup, so the checker PID cannot be reused.
    static_cast<void>(kill(-child, SIGKILL));
    static_cast<void>(kill(child, SIGKILL));
    while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {
    }
    return available;
  }

  Shared* shared_ = nullptr;
};

} // namespace mqt::spank
