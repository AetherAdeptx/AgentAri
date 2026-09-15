#pragma once

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace agentari::diagnostics {

// A deliberately transient run log. Opening a RunLog removes the previous
// file at the exact configured path and starts a new one. Copy a run log
// elsewhere before restarting when a failure investigation needs to keep it.
class RunLog final {
public:
    explicit RunLog(std::filesystem::path path);
    ~RunLog();

    RunLog(const RunLog&) = delete;
    RunLog& operator=(const RunLog&) = delete;

    void info(std::string_view message);
    void debug(std::string_view message);
    void error(std::string_view message);

    [[nodiscard]] bool healthy() const noexcept;
    [[nodiscard]] const std::filesystem::path& path() const noexcept;

private:
    void write(std::string_view level, std::string_view message);
    void writer_loop();

    std::filesystem::path path_;
    std::ofstream stream_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable queue_space_cv_;
    std::vector<std::string> queue_;
    std::size_t queue_head_{0U};
    std::size_t queue_tail_{0U};
    std::size_t queued_count_{0U};
    bool stopping_{false};
    std::thread writer_;
    std::atomic<bool> healthy_{false};
};

}  // namespace agentari::diagnostics
