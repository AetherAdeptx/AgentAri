#pragma once

#include "firstagent/Experience.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace firstagent {

class Memory {
public:
    explicit Memory(std::filesystem::path storage_path);

    bool load(std::string& error);
    bool remember(Experience experience, std::string& error);

    [[nodiscard]] std::vector<Experience> recent(std::size_t limit) const;
    [[nodiscard]] std::vector<Experience> search(const std::string& query) const;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const std::filesystem::path& storage_path() const noexcept;

private:
    std::filesystem::path storage_path_;
    std::vector<Experience> experiences_;

    static std::string escape(const std::string& value);
    static std::string unescape(const std::string& value);
    static std::vector<std::string> split_record(const std::string& line);
};

}  // namespace firstagent
