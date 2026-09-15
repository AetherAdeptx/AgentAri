#include "agentari/RunLog.hpp"

#include <chrono>
#include <functional>
#include <system_error>
#include <thread>
#include <utility>

namespace agentari::diagnostics {

RunLog::RunLog(std::filesystem::path path)
    : path_(std::move(path)) {
    std::error_code filesystem_error;
    if (const std::filesystem::path parent = path_.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent, filesystem_error);
        if (filesystem_error) {
            return;
        }
    }

    // Removing one explicitly resolved file is intentional: this workflow
    // never clears a directory or a collection of unrelated diagnostic data.
    std::filesystem::remove(path_, filesystem_error);
    if (filesystem_error) {
        return;
    }

    stream_.open(path_, std::ios::out | std::ios::trunc);
    if (!stream_.good()) {
        return;
    }

    // Reserve the queue and line storage once. Writers can then enqueue
    // without growing a deque or allocating a node for every diagnostic.
    constexpr std::size_t queue_capacity = 1024U;
    queue_.resize(queue_capacity);
    for (std::string& entry : queue_) {
        entry.reserve(512U);
    }
    try {
        healthy_.store(true, std::memory_order_release);
        writer_ = std::thread(&RunLog::writer_loop, this);
    } catch (...) {
        healthy_.store(false, std::memory_order_release);
        return;
    }
    write("START", "new transient run log opened");
}

RunLog::~RunLog() {
    {
        std::lock_guard lock(queue_mutex_);
        stopping_ = true;
    }
    queue_cv_.notify_one();
    queue_space_cv_.notify_all();
    if (writer_.joinable()) {
        writer_.join();
    }
    if (stream_.is_open()) {
        stream_.close();
    }
}

void RunLog::info(const std::string_view message) {
    write("INFO", message);
}

void RunLog::debug(const std::string_view message) {
    write("DEBUG", message);
}

void RunLog::error(const std::string_view message) {
    write("ERROR", message);
}

bool RunLog::healthy() const noexcept {
    return healthy_.load(std::memory_order_acquire);
}

const std::filesystem::path& RunLog::path() const noexcept {
    return path_;
}

void RunLog::write(const std::string_view level, const std::string_view message) {
    if (!healthy()) {
        return;
    }
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    const auto thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());

    std::string formatted;
    formatted.reserve(64U + level.size() + message.size());
    formatted.append(std::to_string(milliseconds));
    formatted.append(" [");
    formatted.append(level);
    formatted.append("] [thread=");
    formatted.append(std::to_string(thread_id));
    formatted.append("] ");
    formatted.append(message);
    formatted.push_back('\n');

    std::unique_lock lock(queue_mutex_);
    queue_space_cv_.wait(lock, [this] {
        return stopping_ || queued_count_ < queue_.size();
    });
    if (stopping_ || queue_.empty()) {
        return;
    }
    queue_[queue_tail_] = std::move(formatted);
    queue_tail_ = (queue_tail_ + 1U) % queue_.size();
    ++queued_count_;
    lock.unlock();
    queue_cv_.notify_one();
}

void RunLog::writer_loop() {
    for (;;) {
        std::string line;
        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return stopping_ || queued_count_ != 0U;
            });
            if (queued_count_ == 0U && stopping_) {
                return;
            }
            line = std::move(queue_[queue_head_]);
            queue_[queue_head_].clear();
            queue_head_ = (queue_head_ + 1U) % queue_.size();
            --queued_count_;
            lock.unlock();
            queue_space_cv_.notify_one();
        }
        stream_ << line;
        stream_.flush();
        if (!stream_.good()) {
            healthy_.store(false, std::memory_order_release);
        }
    }
}

}  // namespace agentari::diagnostics
