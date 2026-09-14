#include "firstagent/Tool.hpp"

#include <algorithm>
#include <sstream>

namespace firstagent {

bool ToolRegistry::register_tool(Tool tool, std::string& error) {
    if (tool.name.empty() || !tool.function) {
        error = "a tool needs a name and an implementation";
        return false;
    }
    if (contains(tool.name)) {
        error = "tool already registered: " + tool.name;
        return false;
    }
    tools_.push_back(std::move(tool));
    return true;
}

bool ToolRegistry::contains(const std::string& name) const {
    return std::any_of(tools_.begin(), tools_.end(), [&name](const Tool& tool) {
        return tool.name == name;
    });
}

std::string ToolRegistry::execute(const std::string& name,
                                  const std::vector<std::string>& arguments) const {
    for (const auto& tool : tools_) {
        if (tool.name == name) {
            return tool.function(arguments);
        }
    }
    return "unknown tool: " + name;
}

std::string ToolRegistry::describe() const {
    std::ostringstream output;
    output << "Available tools:\n";
    for (const auto& tool : tools_) {
        output << "  " << tool.name << " - " << tool.description << '\n';
    }
    return output.str();
}

}  // namespace firstagent
