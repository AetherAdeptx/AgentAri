#include "agentari/TeacherFeedback.hpp"
#include "agentari/Parallel.hpp"
#include "agentari/RunLog.hpp"
#include "agentari/System.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>

namespace {

using namespace agentari::text::hierarchical;

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " --feedback PATH [--max-records N] [--neurons N] [--workers N] [--log PATH]\n"
              << "\nConsumes the controlled JSONL output of qwen_teacher_judge.py.\n"
              << "The judge model is not loaded by this executable and no checkpoint is written.\n"
              << "--workers 0 uses usable logical CPUs minus two; --log is transient and\n"
              << "the previous file at that exact path is removed at startup.\n";
}

std::size_t skip_space(const std::string& line, std::size_t position) {
    while (position < line.size() &&
           std::isspace(static_cast<unsigned char>(line[position])) != 0) {
        ++position;
    }
    return position;
}

std::optional<std::string> json_string(const std::string& line,
                                       const std::string_view key) {
    const std::string marker = "\"" + std::string(key) + "\"";
    const std::size_t marker_position = line.find(marker);
    if (marker_position == std::string::npos) {
        return std::nullopt;
    }
    std::size_t position = skip_space(line, line.find(':', marker_position + marker.size()));
    if (position == std::string::npos || position >= line.size()) {
        return std::nullopt;
    }
    position = skip_space(line, position + 1U);
    if (position >= line.size() || line[position] != '"') {
        return std::nullopt;
    }
    ++position;
    std::string result;
    while (position < line.size()) {
        const char value = line[position++];
        if (value == '"') {
            return result;
        }
        if (value != '\\' || position >= line.size()) {
            result.push_back(value);
            continue;
        }
        const char escaped = line[position++];
        switch (escaped) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        default:
            // The judge emits UTF-8 directly. Preserve an unfamiliar escape
            // literally so a malformed record cannot silently change text.
            result.push_back('\\');
            result.push_back(escaped);
            break;
        }
    }
    return std::nullopt;
}

std::optional<float> json_number(const std::string& line,
                                 const std::string_view key) {
    const std::string marker = "\"" + std::string(key) + "\"";
    const std::size_t marker_position = line.find(marker);
    if (marker_position == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t colon = line.find(':', marker_position + marker.size());
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t value_position = skip_space(line, colon + 1U);
    if (value_position >= line.size()) {
        return std::nullopt;
    }
    char* end = nullptr;
    const float value = std::strtof(line.c_str() + value_position, &end);
    if (end == line.c_str() + value_position || !std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

float score_or_default(const std::string& line,
                       const std::string_view key,
                       const float fallback) {
    const auto value = json_number(line, key);
    return value.has_value() ? *value : fallback;
}

HierarchicalTokenizer make_tokenizer() {
    return HierarchicalTokenizer(HierarchicalTokenizerConfig{
        .parts = SemanticPartLearnerConfig{
            .capacity = 1024U,
            .minimum_part_length = 2U,
            .maximum_part_length = 16U,
            .minimum_distinct_words = 2U,
            .context_radius = 2U,
        },
        .vocabulary_capacity = 100000U,
        .context = ContextTokenizerConfig{.recent_word_limit = 256U},
    });
}

HierarchicalPredictionNetwork make_network(const std::size_t neurons) {
    PredictionNetworkConfig config;
    config.total_neuron_count = neurons;
    config.neurons_per_layer = 2000U;
    config.additional_layer_count = 24U;
    config.min_neurons_per_layer = Min_Neurons_Per_Layer;
    return HierarchicalPredictionNetwork(config);
}

}  // namespace

int main(int argc, char** argv) {
    std::string feedback_path;
    std::size_t max_records = 0U;
    std::size_t neurons = 56000U;
    std::size_t requested_workers = 0U;
    std::filesystem::path log_path;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto next_value = [&index, argc, argv]() -> std::optional<std::string> {
            if (index + 1 >= argc) {
                return std::nullopt;
            }
            ++index;
            return std::string(argv[index]);
        };
        if (argument == "--feedback") {
            const auto value = next_value();
            if (!value.has_value()) {
                print_usage(argv[0]);
                return 2;
            }
            feedback_path = *value;
        } else if (argument == "--max-records") {
            const auto value = next_value();
            if (!value.has_value()) {
                print_usage(argv[0]);
                return 2;
            }
            max_records = std::stoull(*value);
        } else if (argument == "--neurons") {
            const auto value = next_value();
            if (!value.has_value()) {
                print_usage(argv[0]);
                return 2;
            }
            neurons = std::stoull(*value);
        } else if (argument == "--workers") {
            const auto value = next_value();
            if (!value.has_value()) {
                print_usage(argv[0]);
                return 2;
            }
            requested_workers = std::stoull(*value);
        } else if (argument == "--log") {
            const auto value = next_value();
            if (!value.has_value()) {
                print_usage(argv[0]);
                return 2;
            }
            log_path = *value;
        } else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }
    if (feedback_path.empty() || neurons == 0U) {
        print_usage(argv[0]);
        return 2;
    }

    if (log_path.empty()) {
        const std::filesystem::path feedback_file(feedback_path);
        log_path = feedback_file.has_parent_path()
                       ? feedback_file.parent_path() / "agentari-training.log"
                       : std::filesystem::path("agentari-training.log");
    }
    agentari::parallel::set_default_worker_count(requested_workers);
    agentari::diagnostics::RunLog run_log(log_path);
    const agentari::system::CpuCapabilities capabilities =
        agentari::system::cpu_capabilities();
    run_log.info("startup executable=agentari-teacher-demo feedback=" + feedback_path +
                 " neurons=" + std::to_string(neurons) +
                 " requested_workers=" + std::to_string(requested_workers) +
                 " effective_workers=" +
                 std::to_string(agentari::parallel::default_worker_count()) +
                 " logical_processors=" +
                 std::to_string(capabilities.topology.logical_processors) +
                 " usable_processors=" +
                 std::to_string(capabilities.topology.usable_processors));
    run_log.debug("transient log path=" + log_path.string());

    std::ifstream input(feedback_path);
    if (!input) {
        run_log.error("could not open feedback input");
        std::cerr << "could not open feedback file: " << feedback_path << '\n';
        return 1;
    }

    HierarchicalTokenizer tokenizer = make_tokenizer();
    HierarchicalPredictionNetwork network = make_network(neurons);
    TeacherFeedbackTrainer trainer(tokenizer, network);
    std::size_t records = 0U;
    std::size_t malformed = 0U;
    std::size_t corrections = 0U;
    std::size_t input_line_number = 0U;
    std::string line;
    try {
        while (std::getline(input, line)) {
            ++input_line_number;
            if (max_records != 0U && records >= max_records) {
                break;
            }
            const auto prompt = json_string(line, "prompt");
            const auto candidate = json_string(line, "candidate");
            if (!prompt.has_value() || !candidate.has_value()) {
                ++malformed;
                run_log.info("record_malformed input_line=" +
                             std::to_string(input_line_number) +
                             " malformed_count=" + std::to_string(malformed));
                continue;
            }
            const std::size_t record_number = records + 1U;
            run_log.info("record_begin number=" + std::to_string(record_number) +
                         " input_line=" + std::to_string(input_line_number));
            const HierarchicalTokenTrace prompt_trace = trainer.observe(*prompt);
            const TeacherFeedback feedback{
                .segmentation_quality = score_or_default(line, "segmentation_quality", 0.5F),
                .semantic_quality = score_or_default(line, "semantic_quality", 0.5F),
                .grammar_quality = score_or_default(line, "grammar_quality", 0.5F),
                .context_relevance = score_or_default(line, "context_relevance", 0.5F),
                .confidence = score_or_default(line, "confidence", 0.0F),
                .corrected_text = json_string(line, "corrected_text").value_or(std::string{}),
            };
            const TeacherFeedbackApplication application =
                trainer.apply(prompt_trace, *candidate, feedback);
            records += 1U;
            corrections += application.learned_correction ? 1U : 0U;
            run_log.info("record_complete number=" + std::to_string(records) +
                         " input_line=" + std::to_string(input_line_number) +
                         " steps=" + std::to_string(network.state().step) +
                         " learned_candidate=" +
                         std::string(application.learned_candidate ? "true" : "false") +
                         " learned_correction=" +
                         std::string(application.learned_correction ? "true" : "false"));
            if (!prompt_trace.atomic_tokens.empty()) {
                const auto association = network.inspect_association(
                    PredictionLayer::character, sequence_begin_id,
                    prompt_trace.atomic_tokens.front().id);
                if (association.has_value()) {
                    run_log.info("neuron_snapshot layer=character source=sequence_begin target=" +
                                 std::to_string(prompt_trace.atomic_tokens.front().id) +
                                 " count=" + std::to_string(association->count) +
                                 " weight=" + std::to_string(association->weight) +
                                 " activation=" + std::to_string(association->activation) +
                                 " teacher_bias=" +
                                 std::to_string(association->teacher_bias));
                }
            }
            if (records % 25U == 0U) {
                const PredictionNetworkState progress = network.state();
                run_log.info("progress records=" + std::to_string(records) +
                             " steps=" + std::to_string(progress.step) +
                             " learned_parts=" +
                             std::to_string(tokenizer.parts().size()) +
                             " vocabulary=" +
                             std::to_string(tokenizer.words().size()));
            }
        }
    } catch (const std::exception& exception) {
        run_log.error("fatal exception: " + std::string(exception.what()));
        std::cerr << "teacher feedback run failed: " << exception.what() << '\n';
        return 1;
    }

    const TeacherFeedbackState state = trainer.state();
    const PredictionNetworkState network_state = network.state();
    run_log.info("complete records=" + std::to_string(records) +
                 " malformed=" + std::to_string(malformed) +
                 " corrections=" + std::to_string(corrections) +
                 " steps=" + std::to_string(network_state.step) +
                 " learned_parts=" + std::to_string(tokenizer.parts().size()) +
                 " vocabulary=" + std::to_string(tokenizer.words().size()));
    std::cout << "teacher_feedback records=" << records
              << " malformed=" << malformed
              << " corrections=" << corrections
              << " presented=" << state.presented
              << " applied=" << state.applied
              << " learned_candidates=" << state.learned_candidates
              << " learned_corrections=" << state.learned_corrections
              << " network_steps=" << network_state.step
              << " learned_parts=" << tokenizer.parts().size()
              << " vocabulary=" << tokenizer.words().size() << '\n';
    return 0;
}
