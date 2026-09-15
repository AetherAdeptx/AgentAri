#pragma once

#include "agentari/Memory.hpp"
#include "agentari/Tool.hpp"

#include <string>
#include <string_view>
#include <cstdint>

namespace agentari::text {
class ContextSteering;
class WordPredictor;
}

namespace agentari {

class Agent {
public:
    Agent(Memory& memory,
          const ToolRegistry& tools,
          text::WordPredictor* predictor = nullptr,
          text::ContextSteering* context = nullptr);

    [[nodiscard]] std::string respond(std::string_view input);

private:
    Memory& memory_;
    const ToolRegistry& tools_;
    text::WordPredictor* predictor_{nullptr};
    text::ContextSteering* context_{nullptr};
    std::uint64_t next_log_number_{1U};

    [[nodiscard]] std::string remember(const std::string& original,
                                       const std::string& text);
    [[nodiscard]] std::string recall(const std::string& query) const;
    [[nodiscard]] std::string use_tool(const std::string& original,
                                        const std::string& command);
    [[nodiscard]] std::string observe(const std::string& input);
};

}  // namespace agentari
