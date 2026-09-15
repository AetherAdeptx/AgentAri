#include "agentari/Memory.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace agentari {

Memory::Memory(std::filesystem::path storage_path)
    : storage_path_(std::move(storage_path)) {}

bool Memory::load(std::string& error) {
    experiences_.clear();

    if (!storage_path_.parent_path().empty()) {
        std::error_code filesystem_error;
        std::filesystem::create_directories(storage_path_.parent_path(), filesystem_error);
        if (filesystem_error) {
            error = "could not create memory directory: " + filesystem_error.message();
            return false;
        }
    }

    std::ifstream input(storage_path_);
    if (!input) {
        if (!std::filesystem::exists(storage_path_)) {
            return true;
        }
        error = "could not open memory file: " + storage_path_.string();
        return false;
    }

    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }

        const auto fields = split_record(line);
        if (fields.size() != 4U || (fields[0] != "0" && fields[0] != "1")) {
            error = "invalid memory record on line " + std::to_string(line_number);
            return false;
        }

        experiences_.push_back(Experience{
            .input = unescape(fields[1]),
            .action = unescape(fields[2]),
            .result = unescape(fields[3]),
            .successful = fields[0] == "1",
        });
    }

    return true;
}

bool Memory::remember(Experience experience, std::string& error) {
    if (!storage_path_.parent_path().empty()) {
        std::error_code filesystem_error;
        std::filesystem::create_directories(storage_path_.parent_path(), filesystem_error);
        if (filesystem_error) {
            error = "could not create memory directory: " + filesystem_error.message();
            return false;
        }
    }

    std::ofstream output(storage_path_, std::ios::app);
    if (!output) {
        error = "could not write memory file: " + storage_path_.string();
        return false;
    }

    output << (experience.successful ? "1" : "0") << '\t'
           << escape(experience.input) << '\t'
           << escape(experience.action) << '\t'
           << escape(experience.result) << '\n';

    if (!output) {
        error = "memory write failed: " + storage_path_.string();
        return false;
    }

    experiences_.push_back(std::move(experience));
    return true;
}

std::vector<Experience> Memory::recent(std::size_t limit) const {
    const auto begin = experiences_.size() > limit ? experiences_.size() - limit : 0U;
    return {experiences_.begin() + static_cast<std::ptrdiff_t>(begin), experiences_.end()};
}

std::vector<Experience> Memory::search(const std::string& query) const {
    std::vector<Experience> matches;
    for (const auto& experience : experiences_) {
        if (experience.input.find(query) != std::string::npos ||
            experience.action.find(query) != std::string::npos ||
            experience.result.find(query) != std::string::npos) {
            matches.push_back(experience);
        }
    }
    return matches;
}

std::size_t Memory::size() const noexcept {
    return experiences_.size();
}

const std::filesystem::path& Memory::storage_path() const noexcept {
    return storage_path_;
}

std::string Memory::escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        switch (character) {
        case '\\':
            escaped += "\\\\";
            break;
        case '\t':
            escaped += "\\t";
            break;
        case '\n':
            escaped += "\\n";
            break;
        default:
            escaped += character;
            break;
        }
    }
    return escaped;
}

std::string Memory::unescape(const std::string& value) {
    std::string unescaped;
    unescaped.reserve(value.size());
    bool escaped = false;
    for (const char character : value) {
        if (escaped) {
            switch (character) {
            case 'n':
                unescaped += '\n';
                break;
            case 't':
                unescaped += '\t';
                break;
            default:
                unescaped += character;
                break;
            }
            escaped = false;
        } else if (character == '\\') {
            escaped = true;
        } else {
            unescaped += character;
        }
    }
    if (escaped) {
        unescaped += '\\';
    }
    return unescaped;
}

std::vector<std::string> Memory::split_record(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool escaped = false;
    for (const char character : line) {
        if (!escaped && character == '\t') {
            fields.push_back(std::move(field));
            field.clear();
            continue;
        }
        field += character;
        if (character == '\\' && !escaped) {
            escaped = true;
        } else {
            escaped = false;
        }
    }
    fields.push_back(std::move(field));
    return fields;
}

}  // namespace agentari
