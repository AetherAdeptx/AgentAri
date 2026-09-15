#pragma once

#include <string>

namespace agentari {

struct Experience {
    std::string input;
    std::string action;
    std::string result;
    bool successful{false};
};

}  // namespace agentari
