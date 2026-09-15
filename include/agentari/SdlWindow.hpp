#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <string>

struct SDL_Renderer;
struct SDL_Window;

namespace agentari {

struct WindowState {
    std::chrono::milliseconds uptime{0};
    std::uint64_t ai_ticks{0};
    std::uint64_t rendered_frames{0};
    std::string input;
    std::string status;
};

class SdlWindow {
public:
    SdlWindow() = default;
    ~SdlWindow();

    SdlWindow(const SdlWindow&) = delete;
    SdlWindow& operator=(const SdlWindow&) = delete;

    bool open(std::string& error);
    void close();

    // Process window and keyboard events without blocking.
    bool pump_events(std::deque<std::string>& pending_commands,
                     bool& window_closed,
                     std::string& error);

    [[nodiscard]] const std::string& input_buffer() const noexcept;
    void wait_for_next_tick(std::chrono::milliseconds timeout) const;
    void render(const WindowState& state);

private:
    SDL_Window* window_{nullptr};
    SDL_Renderer* renderer_{nullptr};
    std::string input_buffer_;
};

}  // namespace agentari
