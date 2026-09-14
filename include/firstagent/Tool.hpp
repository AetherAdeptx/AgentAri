#pragma once

#include <functional>
#include <string>
#include <vector>

namespace firstagent {

using ToolFunction = std::function<std::string(const std::vector<std::string>&)>;

struct Tool {
    std::string name;
    std::string description;
    ToolFunction function;
};

class ToolRegistry {
public:
    bool register_tool(Tool tool, std::string& error);
    [[nodiscard]] bool contains(const std::string& name) const;
    [[nodiscard]] std::string execute(const std::string& name,
                                      const std::vector<std::string>& arguments) const;
    [[nodiscard]] std::string describe() const;

private:
    std::vector<Tool> tools_;
};

}  // namespace firstagent
