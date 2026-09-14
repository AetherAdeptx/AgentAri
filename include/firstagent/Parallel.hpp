#pragma once

#include "firstagent/System.hpp"

#include <atomic>
#include <cstddef>
#include <exception>
#include <algorithm>
#include <mutex>
#include <thread>
#include <vector>

namespace firstagent::parallel {

struct Config {
    // Zero means use the configurable process-wide default. The automatic
    // default leaves two logical processors available for the OS/UI.
    std::size_t worker_count{0U};
    std::size_t grain_size{1U};
    // Avoid creating threads for small tensor operations.
    std::size_t minimum_parallel_work{4096U};
};

inline std::atomic<std::size_t> configured_default_worker_count{0U};
inline thread_local bool inside_parallel_worker = false;

[[nodiscard]] inline std::size_t available_worker_count() noexcept {
    const auto& topology = system::cpu_capabilities().topology;
    return std::max<std::size_t>(1U, std::min(topology.logical_processors,
                                              topology.usable_processors));
}

[[nodiscard]] inline std::size_t automatic_worker_count() noexcept {
    const std::size_t available = available_worker_count();
    return available > 2U ? available - 2U : 1U;
}

// A value of zero restores the automatic "logical processors minus two"
// policy. Values above the usable affinity set are capped at that set.
inline void set_default_worker_count(std::size_t worker_count) noexcept {
    configured_default_worker_count.store(worker_count, std::memory_order_release);
}

[[nodiscard]] inline std::size_t default_worker_count() noexcept {
    const std::size_t configured =
        configured_default_worker_count.load(std::memory_order_acquire);
    return std::min(available_worker_count(),
                    configured == 0U ? automatic_worker_count() :
                                       std::max<std::size_t>(1U, configured));
}

inline std::size_t worker_count_for(std::size_t work_items,
                                    const Config& config = {}) noexcept {
    if (work_items == 0U) {
        return 0U;
    }
    const std::size_t available = available_worker_count();
    const std::size_t requested = config.worker_count == 0U
                                      ? default_worker_count()
                                      : config.worker_count;
    const std::size_t grain = std::max<std::size_t>(1U, config.grain_size);
    const std::size_t by_work = work_items / grain +
                                 (work_items % grain == 0U ? 0U : 1U);
    return std::min({available, std::max<std::size_t>(1U, requested), by_work});
}

template <typename Function>
void parallel_for(std::size_t begin,
                  std::size_t end,
                  Function&& function,
                  const Config& config = {}) {
    if (end <= begin) {
        return;
    }
    const std::size_t work_items = end - begin;
    const std::size_t workers = worker_count_for(work_items, config);
    // Nested parallel regions would oversubscribe the machine and can be
    // disastrous when a worker invokes another tensor kernel. Keep nested
    // regions local to the current worker.
    if (inside_parallel_worker || workers <= 1U || work_items < config.minimum_parallel_work) {
        for (std::size_t index = begin; index < end; ++index) {
            function(index);
        }
        return;
    }

    const std::size_t grain = std::max<std::size_t>(1U, config.grain_size);
    std::atomic<std::size_t> next{begin};
    std::atomic<bool> stop{false};
    std::mutex error_mutex;
    std::exception_ptr first_error;

    const auto worker = [&]() {
        const bool previous_context = inside_parallel_worker;
        inside_parallel_worker = true;
        try {
            while (!stop.load(std::memory_order_relaxed)) {
                const std::size_t first = next.fetch_add(grain, std::memory_order_relaxed);
                if (first >= end) {
                    break;
                }
                const std::size_t remaining = end - first;
                const std::size_t last = first + std::min(remaining, grain);
                for (std::size_t index = first; index < last; ++index) {
                    function(index);
                }
            }
        } catch (...) {
            stop.store(true, std::memory_order_relaxed);
            std::lock_guard lock(error_mutex);
            if (first_error == nullptr) {
                first_error = std::current_exception();
            }
        }
        inside_parallel_worker = previous_context;
    };

    std::vector<std::thread> threads;
    threads.reserve(workers - 1U);
    for (std::size_t index = 1U; index < workers; ++index) {
        threads.emplace_back(worker);
    }
    worker();
    for (std::thread& thread : threads) {
        thread.join();
    }
    if (first_error != nullptr) {
        std::rethrow_exception(first_error);
    }
}

}  // namespace firstagent::parallel
