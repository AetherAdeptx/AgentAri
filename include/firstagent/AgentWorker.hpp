#pragma once

#include "firstagent/Agent.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <stop_token>
#include <stdexcept>
#include <string>
#include <thread>

namespace firstagent {

// Owns the blocking agent work on one background thread. The SDL/render loop
// only enqueues input and polls completed responses, so model training or a
// slow tool cannot stall window events.
class AgentWorker {
public:
    struct Result {
        std::uint64_t sequence{0U};
        std::string input;
        std::string response;
    };

    explicit AgentWorker(Agent& agent, std::size_t maximum_pending = 32U);
    ~AgentWorker();

    AgentWorker(const AgentWorker&) = delete;
    AgentWorker& operator=(const AgentWorker&) = delete;

    [[nodiscard]] bool submit(std::string input);
    [[nodiscard]] std::optional<Result> poll();
    [[nodiscard]] std::size_t pending() const;

private:
    struct Request {
        std::uint64_t sequence{0U};
        std::string input;
    };

    Agent& agent_;
    const std::size_t maximum_pending_;
    mutable std::mutex mutex_;
    std::condition_variable_any condition_;
    std::deque<Request> requests_;
    std::deque<Result> results_;
    std::uint64_t next_sequence_{0U};
    std::jthread thread_;

    void run(std::stop_token stop_token);
};

}  // namespace firstagent
