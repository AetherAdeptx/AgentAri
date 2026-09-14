#include "firstagent/SdlWindow.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include <SDL3/SDL.h>

namespace firstagent {
namespace {

void set_color(SDL_Renderer* renderer, SDL_Color color) {
    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
}

void draw_bar(SDL_Renderer* renderer,
              SDL_FRect outline,
              float fill_ratio,
              SDL_Color outline_color,
              SDL_Color fill_color) {
    set_color(renderer, outline_color);
    SDL_RenderRect(renderer, &outline);

    const auto clamped = std::clamp(fill_ratio, 0.0F, 1.0F);
    SDL_FRect fill = outline;
    fill.x += 2.0F;
    fill.y += 2.0F;
    fill.w = static_cast<float>(std::lround((outline.w - 4.0F) * clamped));
    fill.h -= 4.0F;
    if (fill.w > 0 && fill.h > 0) {
        set_color(renderer, fill_color);
        SDL_RenderFillRect(renderer, &fill);
    }
}

std::string one_line(std::string value) {
    std::replace(value.begin(), value.end(), '\n', ' ');
    constexpr std::size_t max_title_length = 100U;
    if (value.size() > max_title_length) {
        value.resize(max_title_length);
        value += "...";
    }
    return value;
}

void erase_last_utf8_code_point(std::string& value) {
    while (!value.empty()) {
        const unsigned char byte = static_cast<unsigned char>(value.back());
        value.pop_back();
        if ((byte & 0xC0U) != 0x80U) {
            break;
        }
    }
}

}  // namespace

SdlWindow::~SdlWindow() {
    close();
}

bool SdlWindow::open(std::string& error) {
    if (window_ != nullptr) {
        return true;
    }

    (void)SDL_SetAppMetadata("FirstAgent", "0.1.0", "org.firstagent.runtime");
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        error = "SDL initialization failed: " + std::string(SDL_GetError());
        return false;
    }

    window_ = SDL_CreateWindow("FirstAgent", 960, 540, SDL_WINDOW_RESIZABLE);
    if (window_ == nullptr) {
        error = "SDL window creation failed: " + std::string(SDL_GetError());
        SDL_Quit();
        return false;
    }

    renderer_ = SDL_CreateRenderer(window_, nullptr);
    if (renderer_ == nullptr) {
        SDL_ClearError();
        renderer_ = SDL_CreateRenderer(window_, "software");
    }
    if (renderer_ == nullptr) {
        error = "SDL renderer creation failed: " + std::string(SDL_GetError());
        close();
        return false;
    }
    SDL_ClearError();

    (void)SDL_StartTextInput(window_);
    return true;
}

void SdlWindow::close() {
    if (window_ != nullptr) {
        (void)SDL_StopTextInput(window_);
    }
    if (renderer_ != nullptr) {
        SDL_DestroyRenderer(renderer_);
        renderer_ = nullptr;
    }
    if (window_ != nullptr) {
        SDL_DestroyWindow(window_);
        window_ = nullptr;
    }
    SDL_Quit();
}

bool SdlWindow::pump_events(std::deque<std::string>& pending_commands,
                            bool& window_closed,
                            std::string& error) {
    window_closed = false;
    SDL_Event event{};
    while (SDL_PollEvent(&event) != 0) {
        switch (event.type) {
        case SDL_EVENT_QUIT:
            window_closed = true;
            break;
        case SDL_EVENT_TEXT_INPUT:
            input_buffer_ += event.text.text;
            break;
        case SDL_EVENT_KEY_DOWN:
            if (event.key.key == SDLK_ESCAPE ||
                (event.key.key == SDLK_Q &&
                 (event.key.mod & SDL_KMOD_CTRL) != 0)) {
                window_closed = true;
            } else if (event.key.key == SDLK_BACKSPACE && !input_buffer_.empty()) {
                erase_last_utf8_code_point(input_buffer_);
            } else if (event.key.key == SDLK_RETURN ||
                       event.key.key == SDLK_KP_ENTER) {
                pending_commands.push_back(std::move(input_buffer_));
                input_buffer_.clear();
            }
            break;
        default:
            break;
        }
    }

    if (SDL_GetError()[0] != '\0') {
        error = "SDL event error: " + std::string(SDL_GetError());
        SDL_ClearError();
        return false;
    }
    return true;
}

const std::string& SdlWindow::input_buffer() const noexcept {
    return input_buffer_;
}

void SdlWindow::wait_for_next_tick(std::chrono::milliseconds timeout) const {
    const auto clamped = std::clamp(timeout.count(), static_cast<std::int64_t>(0),
                                    static_cast<std::int64_t>(1000));
    SDL_Delay(static_cast<std::uint32_t>(clamped));
}

void SdlWindow::render(const WindowState& state) {
    if (renderer_ == nullptr) {
        return;
    }

    int width = 0;
    int height = 0;
    SDL_GetCurrentRenderOutputSize(renderer_, &width, &height);
    set_color(renderer_, SDL_Color{18, 22, 30, 255});
    SDL_RenderClear(renderer_);

    const SDL_FRect panel{40.0F,
                          40.0F,
                          static_cast<float>(std::max(width - 80, 160)),
                          static_cast<float>(std::max(height - 80, 160))};
    set_color(renderer_, SDL_Color{37, 48, 64, 255});
    SDL_RenderFillRect(renderer_, &panel);
    set_color(renderer_, SDL_Color{110, 178, 255, 255});
    SDL_RenderRect(renderer_, &panel);

    const SDL_FRect ai_bar{80.0F, 130.0F, static_cast<float>(std::max(width - 160, 80)), 32.0F};
    const SDL_FRect frame_bar{80.0F,
                              210.0F,
                              static_cast<float>(std::max(width - 160, 80)),
                              32.0F};
    draw_bar(renderer_, ai_bar,
             static_cast<float>(state.ai_ticks % 125U) / 124.0F,
             SDL_Color{130, 150, 180, 255}, SDL_Color{80, 210, 140, 255});
    draw_bar(renderer_, frame_bar,
             static_cast<float>(state.rendered_frames % 60U) / 59.0F,
             SDL_Color{130, 150, 180, 255}, SDL_Color{110, 150, 255, 255});

    const SDL_FRect status_box{80.0F,
                               300.0F,
                               static_cast<float>(std::max(width - 160, 80)),
                               90.0F};
    set_color(renderer_, SDL_Color{26, 34, 46, 255});
    SDL_RenderFillRect(renderer_, &status_box);
    set_color(renderer_, SDL_Color{80, 100, 130, 255});
    SDL_RenderRect(renderer_, &status_box);

    SDL_RenderPresent(renderer_);

    std::ostringstream title;
    title << "FirstAgent | AI: " << state.ai_ticks << " ticks | Frames: "
          << state.rendered_frames << " | " << one_line(state.status);
    if (!state.input.empty()) {
        title << " | Input: " << one_line(state.input);
    }
    SDL_SetWindowTitle(window_, title.str().c_str());
}

}  // namespace firstagent
