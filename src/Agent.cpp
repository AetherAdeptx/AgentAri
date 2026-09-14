#include "firstagent/Agent.hpp"

#include "firstagent/Tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <sstream>

namespace firstagent {
namespace {

std::string trim(std::string value) {
    const auto not_space = [](unsigned char character) {
        return std::isspace(character) == 0;
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::vector<std::string> words(const std::string& value) {
    std::istringstream input(value);
    std::vector<std::string> result;
    std::string word;
    while (input >> word) {
        result.push_back(std::move(word));
    }
    return result;
}

std::string format_experience(const Experience& experience) {
    std::ostringstream output;
    output << "- [" << (experience.successful ? "success" : "recorded") << "] "
           << experience.action << ": " << experience.result;
    return output.str();
}

bool starts_command(const std::string& input, std::string_view command) {
    return input == command ||
           (input.size() > command.size() && input.rfind(command, 0U) == 0U &&
            std::isspace(static_cast<unsigned char>(input[command.size()])) != 0);
}

}  // namespace

Agent::Agent(Memory& memory,
             const ToolRegistry& tools,
             text::WordPredictor* predictor,
             text::ContextSteering* context)
    : memory_(memory), tools_(tools), predictor_(predictor), context_(context) {}

std::string Agent::respond(std::string_view input_view) {
    const std::string input = trim(std::string(input_view));
    if (input.empty()) {
        return "I did not receive an input.";
    }

    if (input == "/help") {
        return "Commands:\n"
               "  /help                         Show this help.\n"
               "  /remember <text>              Store an experience.\n"
               "  /recall [text]                Recall recent or matching experiences.\n"
               "  /use <tool> [arguments...]    Execute an allow-listed tool.\n"
               "  /stats                        Show memory statistics.\n"
               "  /quit                         Exit the program.\n\n" + tools_.describe();
    }

    if (starts_command(input, "/remember")) {
        return remember(input, trim(input.substr(std::string("/remember").size())));
    }

    if (starts_command(input, "/recall")) {
        return recall(trim(input.substr(std::string("/recall").size())));
    }

    if (starts_command(input, "/use")) {
        return use_tool(input, trim(input.substr(std::string("/use").size())));
    }

    if (input == "/stats") {
        return "Experiences stored: " + std::to_string(memory_.size()) +
               "\nMemory file: " + memory_.storage_path().string();
    }

    return observe(input);
}

std::string Agent::remember(const std::string& original, const std::string& text) {
    if (text.empty()) {
        return "Usage: /remember <text>";
    }

    std::string error;
    if (!memory_.remember(Experience{original, "remember", text, true}, error)) {
        return "Memory error: " + error;
    }
    return "Stored experience " + std::to_string(memory_.size()) + ".";
}

std::string Agent::recall(const std::string& query) const {
    const auto experiences = query.empty() ? memory_.recent(5U) : memory_.search(query);
    if (experiences.empty()) {
        return "No matching experiences found.";
    }

    std::ostringstream output;
    output << "Recalled " << experiences.size() << " experience(s):\n";
    for (const auto& experience : experiences) {
        output << format_experience(experience) << '\n';
    }
    return output.str();
}

std::string Agent::use_tool(const std::string& original, const std::string& command) {
    const auto arguments = words(command);
    if (arguments.empty()) {
        return "Usage: /use <tool> [arguments...]";
    }

    const std::string& tool_name = arguments.front();
    if (!tools_.contains(tool_name)) {
        return "Tool denied: " + tool_name + " is not allow-listed.";
    }

    std::vector<std::string> tool_arguments(arguments.begin() + 1, arguments.end());
    std::string result;
    bool successful = true;
    try {
        result = tools_.execute(tool_name, tool_arguments);
    } catch (const std::exception& exception) {
        result = "Tool error: ";
        result += exception.what();
        successful = false;
    } catch (...) {
        result = "Tool error: unknown exception";
        successful = false;
    }
    std::string error;
    if (!memory_.remember(Experience{original, "use:" + tool_name, result, successful}, error)) {
        return result + "\nWarning: result worked, but learning record failed: " + error;
    }
    return result;
}

std::string Agent::observe(const std::string& input) {
    std::string response =
        "I heard: " + input +
        "\nI am a from-scratch learning scaffold; use /remember to teach me an explicit fact.";
    if (predictor_ != nullptr) {
        try {
            predictor_->observe_text(input);
            std::vector<text::Prediction> predictions;
            if (context_ != nullptr) {
                context_->record_chat_text(next_log_number_, input);
                const text::ContextWindow context_window =
                    context_->build_text(next_log_number_, input);
                predictions = predictor_->predict(context_window, 3U);
            } else {
                predictions = predictor_->predict_text(input, 3U);
            }
            if (!predictions.empty()) {
                response += "\nNext-word candidates:";
                for (const text::Prediction& prediction : predictions) {
                    response += " " + prediction.word;
                }
            }
            const text::TokenizerPrimitiveState state = predictor_->tokenizer_state().primitives;
            predictor_->apply_message_feedback(
                std::vector<float>(state.values.begin(), state.values.end()));
        } catch (const std::exception& exception) {
            response += "\nPrediction warning: ";
            response += exception.what();
        } catch (...) {
            response += "\nPrediction warning: unknown predictor failure";
        }
        ++next_log_number_;
    }
    std::string error;
    if (!memory_.remember(Experience{input, "observe", response, false}, error)) {
        return response + "\nWarning: observation was not persisted: " + error;
    }
    return response;
}

}  // namespace firstagent
