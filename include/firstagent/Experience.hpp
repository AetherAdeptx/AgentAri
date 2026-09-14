#pragma once

#include <string>

namespace firstagent {

struct Experience {
    std::string input;
    std::string action;
    std::string result;
    bool successful{false};
};

}  // namespace firstagent
