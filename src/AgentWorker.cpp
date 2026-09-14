#include "firstagent/AgentWorker.hpp"

#include <exception>

namespace firstagent {

AgentWorker::AgentWorker(Agent& agent, std::size_t maximum_pending)
    : agent_(agent), maximum_pending_(maximum_pending == 0U ? throw std::invalid_argument(
          "agent worker queue must have capacity") : maximum_pending),
      thread_([this](std::stop_token stop_token) { run(stop_token); }) {
}

AgentWorker::~AgentWorker() {
    thread_.request_stop();
    condition_.notify_all();
}

bool AgentWorker::submit(std::string input) {
    std::lock_guard lock(mutex_);
    if (requests_.size() >= maximum_pending_) {
        return false;
    }
    requests_.push_back(Request{.sequence = next_sequence_++, .input = std::move(input)});
    condition_.notify_one();
    return true;
}

std::optional<AgentWorker::Result> AgentWorker::poll() {
    std::lock_guard lock(mutex_);
    if (results_.empty()) {
        return std::nullopt;
    }
    Result result = std::move(results_.front());
    results_.pop_front();
    return result;
}

std::size_t AgentWorker::pending() const {
    std::lock_guard lock(mutex_);
    return requests_.size();
}

void AgentWorker::run(std::stop_token stop_token) {
    while (true) {
        Request request;
        {
            std::unique_lock lock(mutex_);
            condition_.wait(lock, stop_token, [this, &stop_token] {
                return stop_token.stop_requested() || !requests_.empty();
            });
            if (requests_.empty()) {
                return;
            }
            request = std::move(requests_.front());
            requests_.pop_front();
        }

        std::string response;
        try {
            response = agent_.respond(request.input);
        } catch (const std::exception& exception) {
            response = "Agent error: ";
            response += exception.what();
        } catch (...) {
            response = "Agent error: unknown exception";
        }

        {
            std::lock_guard lock(mutex_);
            if (results_.size() >= maximum_pending_) {
                results_.pop_front();
            }
            results_.push_back(Result{
                .sequence = request.sequence,
                .input = std::move(request.input),
                .response = std::move(response),
            });
        }
    }
}

}  // namespace firstagent
