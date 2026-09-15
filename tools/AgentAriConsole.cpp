#include "agentari/Hardware.hpp"
#include "agentari/HierarchicalPrediction.hpp"
#include "agentari/Parallel.hpp"
#include "agentari/RunLog.hpp"
#include "agentari/System.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
#include <poll.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace {

using agentari::diagnostics::RunLog;
using agentari::hardware::preferred_simd_backend;
using agentari::text::hierarchical::ContextSnapshot;
using agentari::text::hierarchical::HierarchicalPredictionNetwork;
using agentari::text::hierarchical::HierarchicalTokenTrace;
using agentari::text::hierarchical::HierarchicalTokenizer;
using agentari::text::hierarchical::HierarchicalTokenizerConfig;
using agentari::text::hierarchical::PartSpan;
using agentari::text::hierarchical::PredictionCandidate;
using agentari::text::hierarchical::PredictionNetworkConfig;
using agentari::text::hierarchical::WordId;
using agentari::text::hierarchical::WordTrace;
using agentari::text::hierarchical::WordVocabulary;
using agentari::text::hierarchical::sequence_begin_id;

struct ConsoleOptions {
    std::size_t neurons{56000U};
    std::size_t tokenizer_neurons{8000U};
    std::size_t workers{0U};
    std::size_t max_cpu_usage_percent{100U};
    std::string log_path{"agentari-console.log"};
    std::string state_path{"agentari-console-state.txt"};
    bool resume_state{false};
    bool prompt{true};
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--total-neurons N] [--tokenizer-neurons N]"
                 " [--workers N] [--max-cpu-usage PCT]"
                 " [--log PATH] [--state PATH]"
                 " [--resume-state] [--no-prompt]\n\n"
              << "The default is a fresh 56,000-neuron network: 8,000 tokenizer neurons\n"
              << "plus 48,000 general-layer neurons across 24 outer layers.\n"
              << "Learned records are appended to the state file on each update.\n"
              << "Normal startup deletes only that exact state file; --resume-state loads it.\n\n"
              << "Commands:\n"
              << "  /help                         Show this help\n"
              << "  /learn TEXT                   Learn one text record immediately\n"
              << "  /predict TEXT                 Inspect next char/part/word/context predictions\n"
              << "  /inspect TEXT                 Show lossless token and learned-part decomposition\n"
              << "  /feedback QUALITY CONF TEXT   Apply bounded teacher feedback to existing state\n"
              << "  /state                        Show network, tokenizer, and layer state\n"
              << "  /context                      Show the current recent-word context\n"
              << "  /quit                         End the session\n\n"
              << "Plain text is treated as /learn TEXT. QUALITY and CONF are clamped by the learner.\n";
}

bool parse_size(const std::string_view value, std::size_t& result) {
    if (value.empty()) {
        return false;
    }
    try {
        std::size_t consumed = 0U;
        const unsigned long long parsed = std::stoull(std::string(value), &consumed);
        if (consumed != value.size() ||
            parsed > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
            return false;
        }
        result = static_cast<std::size_t>(parsed);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::string_view trim(std::string_view value) noexcept {
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
        value.remove_prefix(1U);
    }
    while (!value.empty() &&
           (value.back() == ' ' || value.back() == '\t' || value.back() == '\r')) {
        value.remove_suffix(1U);
    }
    return value;
}

bool command_is(std::string_view line, const std::string_view command) noexcept {
    return line == command ||
           (line.size() > command.size() && line.starts_with(command) &&
            (line[command.size()] == ' ' || line[command.size()] == '\t'));
}

std::string_view command_argument(std::string_view line, const std::string_view command) noexcept {
    if (line.size() <= command.size()) {
        return {};
    }
    return trim(line.substr(command.size()));
}

bool remove_exact_state_file(const std::string& path, std::string& error) {
    const std::filesystem::path state_path(path);
    std::error_code filesystem_error;
    if (!std::filesystem::exists(state_path, filesystem_error)) {
        if (filesystem_error) {
            error = "could not inspect state path: " + filesystem_error.message();
            return false;
        }
        return true;
    }
    if (!std::filesystem::is_regular_file(state_path, filesystem_error)) {
        error = filesystem_error
                    ? "could not inspect state path: " + filesystem_error.message()
                    : "state path is not a regular file: " + path;
        return false;
    }
    if (!std::filesystem::remove(state_path, filesystem_error) && filesystem_error) {
        error = "could not remove state file: " + filesystem_error.message();
        return false;
    }
    return true;
}

HierarchicalTokenizer make_tokenizer() {
    return HierarchicalTokenizer(HierarchicalTokenizerConfig{
        .parts = agentari::text::hierarchical::SemanticPartLearnerConfig{
            .capacity = 1024U,
            .minimum_part_length = 2U,
            .maximum_part_length = 16U,
            .minimum_distinct_words = 2U,
            .context_radius = 2U,
        },
        .vocabulary_capacity = 100000U,
        .context = agentari::text::hierarchical::ContextTokenizerConfig{
            .recent_word_limit = 256U,
        },
    });
}

HierarchicalPredictionNetwork make_network(const std::size_t neurons,
                                            const std::size_t tokenizer_neurons) {
    PredictionNetworkConfig config;
    config.total_neuron_count = neurons;
    config.tokenizer_neuron_count = tokenizer_neurons;
    config.neurons_per_layer = 2000U;
    config.additional_layer_count = 24U;
    config.min_neurons_per_layer =
        agentari::text::hierarchical::Min_Neurons_Per_Layer;
    return HierarchicalPredictionNetwork(config);
}

std::string character_label(const PredictionCandidate& candidate) {
    if (candidate.sequence_end) {
        return "<EOS>";
    }
    if (candidate.id <= 255U) {
        const char value = static_cast<char>(candidate.id);
        if (value == ' ') {
            return "<space>";
        }
        if (value == '\n') {
            return "<newline>";
        }
        if (value == '\t') {
            return "<tab>";
        }
        return std::string{"'"} + value + "'";
    }
    return "<id:" + std::to_string(candidate.id) + ">";
}

std::string part_label(const PredictionCandidate& candidate,
                       const HierarchicalTokenizer& tokenizer) {
    if (candidate.sequence_end) {
        return "<EOS>";
    }
    if (candidate.atomic_fallback) {
        return "atomic(" + character_label(candidate) + ")";
    }
    const auto* part = tokenizer.parts().part(candidate.id);
    return part == nullptr ? "<unknown-part>" : part->text;
}

std::string word_label(const PredictionCandidate& candidate,
                       const HierarchicalTokenizer& tokenizer) {
    if (candidate.sequence_end) {
        return "<EOS>";
    }
    const std::string_view word = tokenizer.words().text(candidate.id);
    return word.empty() ? "<unknown-word>" : std::string(word);
}

template <typename Label>
void print_candidates(const std::string_view title,
                      const std::vector<PredictionCandidate>& candidates,
                      Label&& label) {
    std::cout << title;
    if (candidates.empty()) {
        std::cout << " <no learned association>\n";
        return;
    }
    for (const PredictionCandidate& candidate : candidates) {
        std::cout << ' ' << label(candidate) << "[count=" << candidate.count
                  << ", p=" << std::fixed << std::setprecision(3) << candidate.probability
                  << ']';
    }
    std::cout << '\n';
}

void print_word_candidates(const std::vector<PredictionCandidate>& candidates,
                           const HierarchicalTokenizer& tokenizer) {
    std::cout << "  next word candidates (up to 64):\n";
    if (candidates.empty()) {
        std::cout << "    <no learned association>\n";
        return;
    }
    for (std::size_t index = 0U; index < candidates.size(); ++index) {
        const PredictionCandidate& candidate = candidates[index];
        std::cout << "    [" << std::setw(2) << index << "] "
                  << word_label(candidate, tokenizer) << " [count=" << candidate.count
                  << ", p=" << std::fixed << std::setprecision(3)
                  << candidate.probability << "]\n";
    }
}

const char* layer_name(const std::size_t index) noexcept {
    switch (index) {
    case 0U:
        return "character";
    case 1U:
        return "part";
    case 2U:
        return "word";
    case 3U:
        return "context-left";
    default:
        return "outer";
    }
}

WordId last_known_word(const HierarchicalTokenTrace& trace) noexcept {
    for (auto iterator = trace.words.rbegin(); iterator != trace.words.rend(); ++iterator) {
        if (iterator->word != WordVocabulary::unknown_id) {
            return iterator->word;
        }
    }
    return WordVocabulary::unknown_id;
}

const PartSpan* last_learned_part(const HierarchicalTokenTrace& trace) noexcept {
    for (auto word = trace.words.rbegin(); word != trace.words.rend(); ++word) {
        for (auto part = word->parts.spans.rbegin(); part != word->parts.spans.rend(); ++part) {
            if (part->learned) {
                return &*part;
            }
        }
    }
    return nullptr;
}

#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
class RawTerminal final {
public:
    RawTerminal() = default;
    ~RawTerminal() {
        restore();
    }

    RawTerminal(const RawTerminal&) = delete;
    RawTerminal& operator=(const RawTerminal&) = delete;

    bool activate() noexcept {
        if (::isatty(STDIN_FILENO) == 0 || ::tcgetattr(STDIN_FILENO, &original_) != 0) {
            return false;
        }
        termios raw = original_;
        raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
        raw.c_iflag &= static_cast<tcflag_t>(~(IXON | ICRNL));
        // Keep output post-processing enabled.  In particular, SSH terminals
        // need newline-to-CRLF handling so each picker row starts at column zero.
        raw.c_cc[VMIN] = 1U;
        raw.c_cc[VTIME] = 0U;
        if (::tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
            return false;
        }
        active_ = true;
        return true;
    }

    void restore() noexcept {
        if (active_) {
            (void)::tcsetattr(STDIN_FILENO, TCSANOW, &original_);
            active_ = false;
        }
    }

private:
    termios original_{};
    bool active_{false};
};

class AlternateScreen final {
public:
    AlternateScreen() {
        std::cout << "\033[?1049h\033[2J\033[H" << std::flush;
    }
    ~AlternateScreen() {
        std::cout << "\033[?1049l" << std::flush;
    }

    AlternateScreen(const AlternateScreen&) = delete;
    AlternateScreen& operator=(const AlternateScreen&) = delete;
};

enum class PickerKey : std::uint8_t {
    up,
    down,
    next,
    accept,
    cancel,
};

bool read_terminal_byte(char& value, const int timeout_ms) noexcept {
    pollfd descriptor{};
    descriptor.fd = STDIN_FILENO;
    descriptor.events = POLLIN;
    const int ready = ::poll(&descriptor, 1U, timeout_ms);
    if (ready <= 0 || (descriptor.revents & POLLIN) == 0) {
        return false;
    }
    return ::read(STDIN_FILENO, &value, 1U) == 1;
}

PickerKey read_picker_key() noexcept {
    char first = '\0';
    if (!read_terminal_byte(first, -1)) {
        return PickerKey::cancel;
    }
    if (first == '\t') {
        return PickerKey::next;
    }
    if (first == '\r' || first == '\n') {
        return PickerKey::accept;
    }
    if (first == 'q' || first == 'Q' || first == 27) {
        if (first != 27) {
            return PickerKey::cancel;
        }
        char bracket = '\0';
        char direction = '\0';
        if (read_terminal_byte(bracket, 100) && bracket == '[' &&
            read_terminal_byte(direction, 100)) {
            if (direction == 'A') {
                return PickerKey::up;
            }
            if (direction == 'B') {
                return PickerKey::down;
            }
        }
        return PickerKey::cancel;
    }
    if (first == 'w' || first == 'W') {
        return PickerKey::up;
    }
    if (first == 's' || first == 'S') {
        return PickerKey::down;
    }
    return PickerKey::next;
}
#endif

class ConsoleSession final {
public:
    explicit ConsoleSession(ConsoleOptions options)
        : options_(std::move(options)),
          tokenizer_(make_tokenizer()),
          network_(make_network(options_.neurons, options_.tokenizer_neurons)),
          log_(options_.log_path) {}

    int run() {
        if (options_.resume_state) {
            restored_event_count_ = restore_state();
        }
        if (!open_state_file()) {
            std::cerr << "Could not open state file: " << options_.state_path << '\n';
            return 1;
        }
        const auto& cpu = agentari::system::cpu_capabilities();
        const auto& gpu = agentari::system::gpu_capabilities();
        const auto state = network_.state();
        std::cout << "AgentAri SSH console\n"
                  << "  fresh state: " << (restored_event_count_ == 0U ? "yes" : "no (resumed)")
                  << "\n"
                  << "  restored records: " << restored_event_count_ << "\n"
                  << "  neurons: " << state.total_neuron_count
                  << " (tokenizer=" << state.tokenizer_neuron_count
                  << ", general=" << state.layer_neuron_count << ")"
                  << "\n"
                  << "  layers: " << state.layers.size() << " (4 tokenizer + 24 outer)\n"
                  << "  workers: " << agentari::parallel::default_worker_count()
                  << " (requested " << options_.workers << ")\n"
                  << "  max CPU scheduler budget: " << options_.max_cpu_usage_percent << "%\n"
                  << "  CPU SIMD: " << preferred_simd_backend()
                  << "  GPU candidate: " << (gpu.vulkan_candidate ? "yes" : "no") << '\n'
                  << "  logical/usable CPUs: " << cpu.topology.logical_processors << '/'
                  << cpu.topology.usable_processors << "\n"
                  << "  transient log: " << log_.path() << '\n'
                  << "  state file: " << options_.state_path << '\n'
                  << "Type /help for commands. Plain text learns immediately.\n";
        log_.info("record=" + std::to_string(record_number_) +
                  " session_start restored=" + std::to_string(restored_event_count_) +
                  " neurons=" + std::to_string(state.total_neuron_count) +
                  " tokenizer_neurons=" + std::to_string(state.tokenizer_neuron_count) +
                  " layer_neurons=" + std::to_string(state.layer_neuron_count) +
                  " layers=" + std::to_string(state.layers.size()) +
                  " workers=" + std::to_string(agentari::parallel::default_worker_count()));

        std::string line;
        while (true) {
            if (options_.prompt) {
                std::cout << "agentari> " << std::flush;
            }
            if (!std::getline(std::cin, line)) {
                break;
            }
            if (!handle_line(trim(line))) {
                break;
            }
        }
        log_.info("record=" + std::to_string(record_number_) + " session_end");
        state_stream_.flush();
        std::cout << "Session ended. Learned state is stored in " << options_.state_path
                  << ".\n";
        return 0;
    }

private:
    ConsoleOptions options_;
    HierarchicalTokenizer tokenizer_;
    HierarchicalPredictionNetwork network_;
    RunLog log_;
    std::ofstream state_stream_;
    bool state_healthy_{false};
    std::size_t restored_event_count_{0U};
    std::size_t record_number_{0U};

    static constexpr std::string_view state_header = "AGENTARI_CONSOLE_STATE_V1";

    std::size_t restore_state() {
        std::ifstream input(options_.state_path);
        if (!input) {
            return 0U;
        }
        std::string line;
        if (!std::getline(input, line)) {
            return 0U;
        }
        if (line != state_header) {
            throw std::runtime_error("unsupported console state header");
        }

        std::size_t restored = 0U;
        while (std::getline(input, line)) {
            if (line.empty()) {
                continue;
            }
            std::istringstream event{line};
            char kind = '\0';
            event >> kind;
            if (kind == 'L') {
                std::string text;
                if (!(event >> std::quoted(text))) {
                    throw std::runtime_error("invalid learned record in state file");
                }
                (void)network_.train_step(tokenizer_, text);
            } else if (kind == 'F') {
                float quality = 0.0F;
                float confidence = 1.0F;
                std::string text;
                if (!(event >> quality >> confidence >> std::quoted(text)) ||
                    !std::isfinite(quality) || !std::isfinite(confidence)) {
                    throw std::runtime_error("invalid feedback record in state file");
                }
                const HierarchicalTokenTrace trace = tokenizer_.encode(text);
                tokenizer_.apply_teacher_feedback(trace, quality);
                network_.apply_teacher_feedback(trace, quality, confidence);
            } else {
                throw std::runtime_error("unknown record in state file");
            }
            std::string extra;
            if (event >> extra) {
                throw std::runtime_error("trailing data in state record");
            }
            ++record_number_;
            ++restored;
        }
        return restored;
    }

    bool open_state_file() {
        const std::filesystem::path state_path(options_.state_path);
        std::error_code filesystem_error;
        if (const std::filesystem::path parent = state_path.parent_path(); !parent.empty()) {
            std::filesystem::create_directories(parent, filesystem_error);
            if (filesystem_error) {
                return false;
            }
        }
        state_stream_.open(state_path, std::ios::out | std::ios::app);
        if (!state_stream_.good()) {
            return false;
        }
        const auto size = std::filesystem::file_size(state_path, filesystem_error);
        if (filesystem_error) {
            return false;
        }
        if (size == 0U) {
            state_stream_ << state_header << '\n';
            state_stream_.flush();
        }
        state_healthy_ = state_stream_.good();
        return state_healthy_;
    }

    bool append_learn_event(const std::string_view text) {
        if (!state_healthy_) {
            return false;
        }
        state_stream_ << "L " << std::quoted(std::string(text)) << '\n';
        state_stream_.flush();
        state_healthy_ = state_stream_.good();
        return state_healthy_;
    }

    bool append_feedback_event(const float quality,
                               const float confidence,
                               const std::string_view text) {
        if (!state_healthy_) {
            return false;
        }
        state_stream_ << "F " << std::setprecision(9) << quality << ' ' << confidence << ' '
                      << std::quoted(std::string(text)) << '\n';
        state_stream_.flush();
        state_healthy_ = state_stream_.good();
        return state_healthy_;
    }

    void report_state_write_failure() const {
        std::cout << "  warning: learned state could not be written to "
                  << options_.state_path << '\n';
    }

    bool handle_line(const std::string_view line) {
        if (line.empty()) {
            return true;
        }
        if (line == "/quit" || line == "/exit") {
            return false;
        }
        if (line == "/help") {
            print_usage("agentari-console");
            return true;
        }
        if (line == "/state") {
            print_state();
            return true;
        }
        if (line == "/context") {
            print_context();
            return true;
        }
        if (command_is(line, "/learn")) {
            return learn(command_argument(line, "/learn"));
        }
        if (command_is(line, "/predict")) {
            return predict(command_argument(line, "/predict"));
        }
        if (command_is(line, "/inspect")) {
            return inspect(command_argument(line, "/inspect"));
        }
        if (command_is(line, "/feedback")) {
            return feedback(command_argument(line, "/feedback"));
        }
        if (line.front() == '/') {
            std::cout << "Unknown command. Type /help.\n";
            log_.error("record=" + std::to_string(record_number_) + " unknown_command");
            return true;
        }
        return learn(line);
    }

    bool learn(const std::string_view text) {
        if (text.empty()) {
            std::cout << "Usage: /learn TEXT\n";
            return true;
        }
        try {
            const auto started = std::chrono::steady_clock::now();
            const HierarchicalTokenTrace trace = network_.train_step(tokenizer_, text);
            ++record_number_;
            if (!append_learn_event(text)) {
                report_state_write_failure();
            }
            const auto elapsed = std::chrono::duration<double>(
                                      std::chrono::steady_clock::now() - started)
                                      .count();
            const auto state = network_.state();
            std::cout << "learned record=" << record_number_ << " step=" << state.step
                      << " bytes=" << text.size() << " atomic=" << trace.atomic_tokens.size()
                      << " words=" << trace.words.size() << " parts=" << tokenizer_.parts().size()
                      << " vocabulary=" << tokenizer_.words().size() << " elapsed_s="
                      << std::fixed << std::setprecision(4) << elapsed << '\n';
            log_.info("record=" + std::to_string(record_number_) +
                      " learned bytes=" + std::to_string(text.size()) +
                      " atomic=" + std::to_string(trace.atomic_tokens.size()) +
                      " words=" + std::to_string(trace.words.size()) +
                      " step=" + std::to_string(state.step));
            print_predictions(trace, false);
        } catch (const std::exception& exception) {
            std::cout << "learn failed: " << exception.what() << '\n';
            log_.error("record=" + std::to_string(record_number_) +
                       " learn_failed=" + exception.what());
        }
        return true;
    }

    bool predict(const std::string_view text) const {
        if (text.empty()) {
            std::cout << "Usage: /predict TEXT\n";
            return true;
        }
        const HierarchicalTokenTrace trace = tokenizer_.encode(text);
        std::cout << "read-only prediction trace: bytes=" << text.size()
                  << " atomic=" << trace.atomic_tokens.size()
                  << " words=" << trace.words.size() << '\n';
        print_predictions(trace, true);
        return true;
    }

    bool inspect(const std::string_view text) const {
        if (text.empty()) {
            std::cout << "Usage: /inspect TEXT\n";
            return true;
        }
        const HierarchicalTokenTrace trace = tokenizer_.encode(text);
        std::cout << "inspection: atomic_tokens=" << trace.atomic_tokens.size()
                  << " words=" << trace.words.size() << '\n';
        const std::size_t word_limit = std::min<std::size_t>(trace.words.size(), 32U);
        for (std::size_t index = 0U; index < word_limit; ++index) {
            const WordTrace& word = trace.words[index];
            std::cout << "  word[" << index << "] span=" << word.span.begin << ':'
                      << word.span.end << " text=\"" << word.span.text << "\" id=";
            if (word.word == WordVocabulary::unknown_id) {
                std::cout << "<unknown>";
            } else {
                std::cout << word.word;
            }
            std::cout << " parts=" << word.parts.spans.size()
                      << " coverage=" << std::fixed << std::setprecision(3)
                      << word.parts.learned_coverage << " [";
            const std::size_t part_limit = std::min<std::size_t>(word.parts.spans.size(), 16U);
            for (std::size_t part_index = 0U; part_index < part_limit; ++part_index) {
                if (part_index != 0U) {
                    std::cout << ", ";
                }
                const PartSpan& part = word.parts.spans[part_index];
                const std::size_t begin = std::min(part.begin, word.span.text.size());
                const std::size_t end = std::min(part.end, word.span.text.size());
                std::cout << (part.learned ? "learned:" : "atomic:")
                          << word.span.text.substr(begin, end - begin);
            }
            if (word.parts.spans.size() > part_limit) {
                std::cout << ", ...";
            }
            std::cout << "]\n";
        }
        if (trace.words.size() > word_limit) {
            std::cout << "  ... " << trace.words.size() - word_limit
                      << " additional words omitted\n";
        }
        return true;
    }

    bool feedback(const std::string_view arguments) {
        std::istringstream input{std::string(arguments)};
        float quality = 0.0F;
        float confidence = 1.0F;
        if (!(input >> quality >> confidence)) {
            std::cout << "Usage: /feedback QUALITY CONF TEXT\n";
            return true;
        }
        std::string text;
        std::getline(input, text);
        const std::string_view trimmed_text = trim(text);
        if (trimmed_text.empty() || !std::isfinite(quality) || !std::isfinite(confidence)) {
            std::cout << "Feedback requires finite QUALITY, CONF, and non-empty TEXT.\n";
            return true;
        }
        const HierarchicalTokenTrace trace = tokenizer_.encode(trimmed_text);
        tokenizer_.apply_teacher_feedback(trace, quality);
        network_.apply_teacher_feedback(trace, quality, confidence);
        ++record_number_;
        if (!append_feedback_event(quality, confidence, trimmed_text)) {
            report_state_write_failure();
        }
        std::cout << "feedback applied record=" << record_number_ << " quality="
                  << std::fixed << std::setprecision(3) << quality << " confidence="
                  << confidence << " (existing associations only)\n";
        log_.info("record=" + std::to_string(record_number_) + " feedback quality=" +
                  std::to_string(quality) + " confidence=" + std::to_string(confidence));
        return true;
    }

    void print_predictions(const HierarchicalTokenTrace& trace,
                           const bool allow_picker) const {
        const std::uint32_t current_character = trace.atomic_tokens.empty()
                                                    ? sequence_begin_id
                                                    : trace.atomic_tokens.back().id;
        print_candidates("  next character:",
                         network_.predict_next_character(current_character, 5U),
                         [](const PredictionCandidate& candidate) {
                             return character_label(candidate);
                         });

        const PartSpan* part = last_learned_part(trace);
        if (part != nullptr) {
            print_candidates("  next part:", network_.predict_next_part(part->id, 5U),
                             [this](const PredictionCandidate& candidate) {
                                 return part_label(candidate, tokenizer_);
                             });
        } else {
            std::cout << "  next part: <no learned part in trace>\n";
        }

        const WordId word = last_known_word(trace);
        const auto word_candidates = network_.predict_next_word(
            word == WordVocabulary::unknown_id ? sequence_begin_id : word,
            allow_picker ? 64U : 5U);
        if (allow_picker) {
            print_word_candidates(word_candidates, tokenizer_);
            if (picker_available() && !word_candidates.empty()) {
                const std::optional<std::size_t> selected = pick_word(word_candidates);
                if (selected.has_value()) {
                    const PredictionCandidate& candidate = word_candidates[*selected];
                    std::cout << "  selected word: " << word_label(candidate, tokenizer_)
                              << " (index " << *selected << ")\n";
                } else {
                    std::cout << "  word selection cancelled\n";
                }
            }
        } else {
            print_candidates("  next word:", word_candidates,
                             [this](const PredictionCandidate& candidate) {
                                 return word_label(candidate, tokenizer_);
                             });
        }
        if (word == WordVocabulary::unknown_id) {
            std::cout << "  left context: <no known current word>\n";
        } else {
            print_candidates("  left context:", network_.predict_left_context(word, 5U),
                             [this](const PredictionCandidate& candidate) {
                                 return word_label(candidate, tokenizer_);
                             });
        }
    }

    bool picker_available() const noexcept {
#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
        return options_.prompt && ::isatty(STDIN_FILENO) != 0 && ::isatty(STDOUT_FILENO) != 0;
#else
        return false;
#endif
    }

    std::optional<std::size_t> pick_word(
        const std::vector<PredictionCandidate>& candidates) const {
#if defined(__linux__) || defined(__APPLE__) || defined(__unix__)
        RawTerminal terminal;
        if (!terminal.activate()) {
            return std::nullopt;
        }
        AlternateScreen screen;
        std::size_t selected = 0U;
        for (;;) {
            std::cout << "\033[2J\033[H"
                      << "AgentAri word picker (" << candidates.size() << " candidates)\n"
                      << "  Up/Down: move   Tab: next   Enter: select   Esc/Q: cancel\n\n";
            for (std::size_t index = 0U; index < candidates.size(); ++index) {
                const PredictionCandidate& candidate = candidates[index];
                std::cout << (index == selected ? "  > " : "    ") << '[' << std::setw(2)
                          << index << "] " << word_label(candidate, tokenizer_)
                          << "  count=" << candidate.count << " p=" << std::fixed
                          << std::setprecision(3) << candidate.probability << '\n';
            }
            std::cout << std::flush;

            switch (read_picker_key()) {
            case PickerKey::up:
                selected = selected == 0U ? candidates.size() - 1U : selected - 1U;
                break;
            case PickerKey::down:
            case PickerKey::next:
                selected = (selected + 1U) % candidates.size();
                break;
            case PickerKey::accept:
                return selected;
            case PickerKey::cancel:
                return std::nullopt;
            }
        }
#else
        (void)candidates;
        return std::nullopt;
#endif
    }

    void print_state() const {
        const auto state = network_.state();
        std::uint64_t total_observations = 0U;
        std::uint64_t total_transitions = 0U;
        for (const auto& layer : state.layers) {
            total_observations += layer.observations;
            total_transitions += layer.transition_neurons;
        }
        std::cout << "state: step=" << state.step << " neurons=" << state.total_neuron_count
                  << " tokenizer_neurons=" << state.tokenizer_neuron_count
                  << " layer_neurons=" << state.layer_neuron_count
                  << " objects=" << network_.instantiated_neuron_count()
                  << " layers=" << state.layers.size()
                  << " observations=" << total_observations
                  << " transition_neurons=" << total_transitions
                  << " learned_parts=" << tokenizer_.parts().size()
                  << " vocabulary=" << tokenizer_.words().size() << '\n';
        const auto& mix = network_.neuron_mix_allocation();
        std::cout << "  neuron mix:";
        for (std::size_t index = 0U; index < mix.entry_count; ++index) {
            const auto& entry = mix.entries[index];
            const std::string_view type_name = entry.type_name.empty()
                                                   ? agentari::neuron::neuron_type_name(entry.type)
                                                   : std::string_view(entry.type_name);
            std::cout << ' ' << type_name << '='
                      << entry.neuron_count;
            if (entry.mode == agentari::neuron::NeuronAllocationMode::fixed_count) {
                std::cout << "[fixed]";
            } else {
                std::cout << '(' << entry.fill_percent << "%)";
            }
            if (entry.layer_id != agentari::neuron::neuron_any_layer) {
                std::cout << "@L" << entry.layer_id;
            }
        }
        std::cout << " assigned=" << mix.assigned_neurons
                  << " unassigned=" << mix.unassigned_neurons << '\n';
        const auto metrics = network_.spatial_map().distribution_metrics();
        std::cout << "  spatial: occupied_cells="
                  << network_.spatial_map().occupied_cell_count()
                  << " ideal_spacing=" << metrics.ideal_isometric_spacing
                  << " average_nearest_neighbor="
                  << metrics.average_nearest_neighbor_distance << '\n';
        for (std::size_t index = 0U; index < state.layers.size(); ++index) {
            const auto& layer = state.layers[index];
            std::cout << "  " << std::setw(11) << std::left << layer_name(index)
                      << " neurons=" << std::setw(6) << std::right << layer.neuron_count
                      << " local=" << std::setw(6) << layer.local_input_neurons
                      << " outer=" << std::setw(6) << layer.network_reference_neurons
                      << " cross=" << std::setw(6) << layer.cross_layer_input_neurons
                      << " obs=" << layer.observations
                      << " transitions=" << layer.transition_neurons << '\n';
        }
    }

    void print_context() const {
        const ContextSnapshot context = tokenizer_.context().snapshot();
        std::cout << "context: observed_words=" << context.observed_words
                  << " retained_words=" << context.recent_words.size() << "\n";
        const std::size_t begin = context.recent_words.size() > 16U
                                      ? context.recent_words.size() - 16U
                                      : 0U;
        std::cout << "  recent:";
        for (std::size_t index = begin; index < context.recent_words.size(); ++index) {
            const std::string_view word = tokenizer_.words().text(context.recent_words[index]);
            std::cout << ' ' << (word.empty() ? "<unknown>" : word);
        }
        std::cout << '\n';
    }
};

}  // namespace

int main(int argc, char* argv[]) {
    ConsoleOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        if (argument == "--no-prompt") {
            options.prompt = false;
            continue;
        }
        if (argument == "--resume-state") {
            options.resume_state = true;
            continue;
        }
        if (argument == "--neurons" || argument == "--total-neurons" ||
            argument == "--tokenizer-neurons" ||
            argument == "--tokenizer-neuron-count" ||
            argument == "--workers" || argument == "--max-cpu-usage" ||
            argument == "--log" || argument == "--state") {
            if (index + 1 >= argc) {
                std::cerr << "Missing value for " << argument << '\n';
                return 2;
            }
            const std::string_view value(argv[++index]);
            if (argument == "--log" || argument == "--state") {
                if (value.empty()) {
                    std::cerr << argument << " requires a path\n";
                    return 2;
                }
                if (argument == "--log") {
                    options.log_path = std::string(value);
                } else {
                    options.state_path = std::string(value);
                }
                continue;
            }
            std::size_t parsed = 0U;
            if (!parse_size(value, parsed) ||
                ((argument == "--neurons" || argument == "--total-neurons" ||
                  argument == "--tokenizer-neurons" ||
                  argument == "--tokenizer-neuron-count") &&
                 parsed == 0U) ||
                (argument == "--max-cpu-usage" && (parsed == 0U || parsed > 100U))) {
                std::cerr << "Invalid value for " << argument << ": " << value << '\n';
                return 2;
            }
            if (argument == "--neurons" || argument == "--total-neurons") {
                options.neurons = parsed;
            } else if (argument == "--tokenizer-neurons" ||
                       argument == "--tokenizer-neuron-count") {
                options.tokenizer_neurons = parsed;
            } else if (argument == "--max-cpu-usage") {
                options.max_cpu_usage_percent = parsed;
            } else {
                options.workers = parsed;
            }
            continue;
        }
        print_usage(argv[0]);
        return 2;
    }

    if (!options.resume_state) {
        std::string state_error;
        if (!remove_exact_state_file(options.state_path, state_error)) {
            std::cerr << "Could not clear prior state: " << state_error << '\n';
            return 2;
        }
    }
    agentari::parallel::set_max_cpu_usage_percent(options.max_cpu_usage_percent);
    agentari::parallel::set_default_worker_count(options.workers);
    try {
        return ConsoleSession(options).run();
    } catch (const std::exception& exception) {
        std::cerr << "AgentAri console failed: " << exception.what() << '\n';
        return 1;
    }
}
