#pragma once

#include "agentari/System.hpp"

#include <atomic>
#include <cstddef>
#include <exception>
#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace agentari::parallel {

struct Config {
    // Zero means use the configurable process-wide default. The automatic
    // default leaves two logical processors available for the OS/UI.
    std::size_t worker_count{0U};
    std::size_t grain_size{1U};
    // Avoid creating threads for small tensor operations.
    std::size_t minimum_parallel_work{4096U};
};

inline std::atomic<std::size_t> configured_default_worker_count{0U};
inline std::atomic<std::size_t> configured_max_cpu_usage_percent{100U};
inline thread_local bool inside_parallel_worker = false;

namespace detail {

// One invocation is stack-owned by the caller. The pool only retains a
// pointer until all workers have completed, so parallel operations do not
// allocate a task object on every call.
struct ParallelInvocation {
    using Callback = void (*)(void*, std::size_t);

    Callback callback{nullptr};
    void* context{nullptr};
    std::size_t end{0U};
    std::size_t grain{1U};
    std::atomic<std::size_t> next{0U};
    std::atomic<std::size_t> remaining{0U};
    std::atomic<bool> stop{false};
    std::mutex error_mutex;
    std::exception_ptr first_error;
};

class PersistentWorkerPool final {
public:
    static PersistentWorkerPool& instance() {
        static PersistentWorkerPool pool;
        return pool;
    }

    PersistentWorkerPool(const PersistentWorkerPool&) = delete;
    PersistentWorkerPool& operator=(const PersistentWorkerPool&) = delete;

    void run(ParallelInvocation& invocation, const std::size_t worker_count) {
        ensure_workers(worker_count > 0U ? worker_count - 1U : 0U);
        invocation.remaining.store(worker_count, std::memory_order_release);
        {
            std::lock_guard lock(work_mutex_);
            invocation_ = &invocation;
            active_worker_count_ = worker_count - 1U;
            ++generation_;
        }
        work_cv_.notify_all();

        execute(invocation);
        for (;;) {
            const std::size_t remaining =
                invocation.remaining.load(std::memory_order_acquire);
            if (remaining == 0U) {
                break;
            }
            invocation.remaining.wait(remaining, std::memory_order_acquire);
        }

        {
            std::lock_guard lock(work_mutex_);
            invocation_ = nullptr;
            active_worker_count_ = 0U;
        }
        work_cv_.notify_all();
        if (invocation.first_error != nullptr) {
            std::rethrow_exception(invocation.first_error);
        }
    }

private:
    PersistentWorkerPool() = default;

    ~PersistentWorkerPool() {
        {
            std::lock_guard lock(work_mutex_);
            stopping_ = true;
        }
        work_cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    void ensure_workers(const std::size_t required) {
        std::lock_guard lock(work_mutex_);
        while (workers_.size() < required) {
            const std::size_t index = workers_.size();
            workers_.emplace_back([this, index] { worker_loop(index); });
        }
    }

    static void execute(ParallelInvocation& invocation) {
        try {
            while (!invocation.stop.load(std::memory_order_relaxed)) {
                const std::size_t first =
                    invocation.next.fetch_add(invocation.grain, std::memory_order_relaxed);
                if (first >= invocation.end) {
                    break;
                }
                const std::size_t remaining = invocation.end - first;
                const std::size_t last = first +
                    std::min(remaining, invocation.grain);
                for (std::size_t index = first; index < last; ++index) {
                    invocation.callback(invocation.context, index);
                }
            }
        } catch (...) {
            invocation.stop.store(true, std::memory_order_relaxed);
            std::lock_guard lock(invocation.error_mutex);
            if (invocation.first_error == nullptr) {
                invocation.first_error = std::current_exception();
            }
        }
        if (invocation.remaining.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
            invocation.remaining.notify_all();
        }
    }

    void worker_loop(const std::size_t index) {
        std::uint64_t seen_generation = 0U;
        for (;;) {
            ParallelInvocation* invocation = nullptr;
            std::uint64_t generation = 0U;
            {
                std::unique_lock lock(work_mutex_);
                work_cv_.wait(lock, [this, index, seen_generation] {
                    return stopping_ ||
                           (invocation_ != nullptr && index < active_worker_count_ &&
                            generation_ != seen_generation);
                });
                if (stopping_) {
                    return;
                }
                invocation = invocation_;
                generation = generation_;
            }
            seen_generation = generation;
            const bool previous_context = inside_parallel_worker;
            inside_parallel_worker = true;
            execute(*invocation);
            inside_parallel_worker = previous_context;
        }
    }

    std::mutex work_mutex_;
    std::condition_variable work_cv_;
    std::vector<std::thread> workers_;
    ParallelInvocation* invocation_{nullptr};
    std::size_t active_worker_count_{0U};
    std::uint64_t generation_{0U};
    bool stopping_{false};
};

}  // namespace detail

[[nodiscard]] inline std::size_t available_worker_count() noexcept {
    const auto& topology = system::cpu_capabilities().topology;
    const std::size_t available = std::max<std::size_t>(
        1U, std::min(topology.logical_processors, topology.usable_processors));
    const std::size_t usage = std::clamp(
        configured_max_cpu_usage_percent.load(std::memory_order_acquire),
        std::size_t{1U}, std::size_t{100U});
    const std::size_t limited =
        (available * usage + 99U) / 100U;
    return std::max<std::size_t>(1U, std::min(available, limited));
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

// This is a scheduling budget, not a measurement of instantaneous package
// utilization. It caps the number of worker threads selected by the default
// policy while keeping at least one worker available for progress.
inline void set_max_cpu_usage_percent(const std::size_t percent) noexcept {
    configured_max_cpu_usage_percent.store(
        std::clamp(percent, std::size_t{1U}, std::size_t{100U}),
        std::memory_order_release);
}

[[nodiscard]] inline std::size_t max_cpu_usage_percent() noexcept {
    return std::clamp(
        configured_max_cpu_usage_percent.load(std::memory_order_acquire),
        std::size_t{1U}, std::size_t{100U});
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
    struct CallbackAdapter {
        using FunctionType = std::remove_reference_t<Function>;
        FunctionType* function;
        static void invoke(void* context, const std::size_t index) {
            auto* adapter = static_cast<CallbackAdapter*>(context);
            (*adapter->function)(index);
        }
    } adapter{&function};

    detail::ParallelInvocation invocation;
    invocation.callback = &CallbackAdapter::invoke;
    invocation.context = &adapter;
    invocation.end = end;
    invocation.grain = grain;
    invocation.next.store(begin, std::memory_order_relaxed);
    const bool previous_context = inside_parallel_worker;
    inside_parallel_worker = true;
    try {
        detail::PersistentWorkerPool::instance().run(invocation, workers);
    } catch (...) {
        inside_parallel_worker = previous_context;
        throw;
    }
    inside_parallel_worker = previous_context;
}

}  // namespace agentari::parallel
