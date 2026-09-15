#include "agentari/HierarchicalPrediction.hpp"

#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using agentari::text::hierarchical::HierarchicalPredictionNetwork;
using agentari::text::hierarchical::HierarchicalTokenTrace;
using agentari::text::hierarchical::HierarchicalTokenizer;
using agentari::text::hierarchical::HierarchicalTokenizerConfig;
using agentari::text::hierarchical::PartId;
using agentari::text::hierarchical::PredictionCandidate;
using agentari::text::hierarchical::PredictionNetworkConfig;
using agentari::text::hierarchical::sequence_begin_id;

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " [--train PATH] [--eval PATH] [--passes N] [--batch-size N]"
                 " [--max-records N] [--progress-every N] [--prompt TEXT]\n"
              << "\nWithout --train, a small built-in corpus is repeated in memory.\n"
              << "Input records are trained online in bounded batches.\n"
              << "No checkpoint is read or written.\n";
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
                  << ", p=" << candidate.probability << ']';
    }
    std::cout << '\n';
}

struct TrainingOptions {
    std::size_t passes{8U};
    std::size_t batch_size{64U};
    std::size_t max_records{0U};
    std::size_t progress_every{100U};
};

struct PredictionMetrics {
    std::size_t records{0U};
    std::size_t batches{0U};
    std::uint64_t word_targets{0U};
    std::uint64_t word_top1_hits{0U};
};

void score_word_trace(const HierarchicalTokenTrace& trace,
                      const HierarchicalPredictionNetwork& network,
                      PredictionMetrics& metrics) {
    std::vector<agentari::text::hierarchical::WordId> known_words;
    known_words.reserve(trace.words.size());
    for (const auto& word : trace.words) {
        if (word.word != agentari::text::hierarchical::WordVocabulary::unknown_id) {
            known_words.push_back(word.word);
        }
    }
    if (known_words.empty()) {
        return;
    }

    const auto score = [&network, &metrics](const std::uint32_t source,
                                             const std::uint32_t target) {
        ++metrics.word_targets;
        const auto candidates = network.predict_next_word(source, 1U);
        if (!candidates.empty() && !candidates.front().sequence_end &&
            candidates.front().id == target) {
            ++metrics.word_top1_hits;
        }
    };
    score(sequence_begin_id, known_words.front());
    for (std::size_t index = 1U; index < known_words.size(); ++index) {
        score(known_words[index - 1U], known_words[index]);
    }
}

void flush_batch(HierarchicalPredictionNetwork& network,
                 std::vector<HierarchicalTokenTrace>& batch,
                 PredictionMetrics& metrics) {
    if (batch.empty()) {
        return;
    }
    for (const HierarchicalTokenTrace& trace : batch) {
        score_word_trace(trace, network, metrics);
    }
    network.observe_batch(std::span<const HierarchicalTokenTrace>(batch.data(), batch.size()));
    ++metrics.batches;
    batch.clear();
}

void print_progress(const PredictionMetrics& metrics,
                    const TrainingOptions& options,
                    const std::chrono::steady_clock::time_point started) {
    if (options.progress_every == 0U || metrics.records == 0U ||
        metrics.records % options.progress_every != 0U) {
        return;
    }
    const auto elapsed = std::chrono::duration<float>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    const float accuracy = metrics.word_targets == 0U
                               ? 0.0F
                               : 100.0F * static_cast<float>(metrics.word_top1_hits) /
                                     static_cast<float>(metrics.word_targets);
    std::cerr << "progress records=" << metrics.records
              << " batches=" << metrics.batches << " word_top1=" << std::fixed
              << std::setprecision(2) << accuracy << "% elapsed_s=" << std::setprecision(1)
              << elapsed << '\n';
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

HierarchicalPredictionNetwork make_network() {
    PredictionNetworkConfig config;
    config.total_neuron_count = 0U;
    config.neurons_per_layer = 2000U;
    config.additional_layer_count = 24U;
    config.min_neurons_per_layer =
        agentari::text::hierarchical::Min_Neurons_Per_Layer;
    return HierarchicalPredictionNetwork(config);
}

PredictionMetrics train_builtin(HierarchicalTokenizer& tokenizer,
                                HierarchicalPredictionNetwork& network,
                                const TrainingOptions& options) {
    const std::vector<std::string> corpus{
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox watches the dog",
        "the agent learns from text and predicts the next word",
        "an agent learns while the network observes text",
        "unhappy happyish unkind darkness",
    };
    PredictionMetrics metrics;
    std::vector<HierarchicalTokenTrace> batch;
    batch.reserve(options.batch_size);
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t pass = 0U; pass < options.passes; ++pass) {
        for (const std::string& line : corpus) {
            if (options.max_records != 0U && metrics.records >= options.max_records) {
                break;
            }
            batch.push_back(tokenizer.observe(line));
            ++metrics.records;
            if (batch.size() >= options.batch_size) {
                flush_batch(network, batch, metrics);
            }
            print_progress(metrics, options, started);
        }
        if (options.max_records != 0U && metrics.records >= options.max_records) {
            break;
        }
    }
    flush_batch(network, batch, metrics);
    return metrics;
}

PredictionMetrics train_file(HierarchicalTokenizer& tokenizer,
                             HierarchicalPredictionNetwork& network,
                             const std::string& path,
                             const TrainingOptions& options) {
    PredictionMetrics metrics;
    std::vector<HierarchicalTokenTrace> batch;
    batch.reserve(options.batch_size);
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t pass = 0U; pass < options.passes; ++pass) {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("could not open training file: " + path);
        }
        std::string line;
        while (std::getline(input, line) &&
               (options.max_records == 0U || metrics.records < options.max_records)) {
            if (line.empty()) {
                continue;
            }
            batch.push_back(tokenizer.observe(line));
            ++metrics.records;
            if (batch.size() >= options.batch_size) {
                flush_batch(network, batch, metrics);
            }
            print_progress(metrics, options, started);
        }
        if (options.max_records != 0U && metrics.records >= options.max_records) {
            break;
        }
    }
    flush_batch(network, batch, metrics);
    return metrics;
}

PredictionMetrics evaluate_file(const HierarchicalTokenizer& tokenizer,
                                const HierarchicalPredictionNetwork& network,
                                const std::string& path,
                                const std::size_t max_records) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("could not open evaluation file: " + path);
    }
    PredictionMetrics metrics;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || (max_records != 0U && metrics.records >= max_records)) {
            continue;
        }
        const HierarchicalTokenTrace trace = tokenizer.encode(line);
        score_word_trace(trace, network, metrics);
        ++metrics.records;
    }
    return metrics;
}

void print_metrics(const std::string_view label, const PredictionMetrics& metrics) {
    const float accuracy = metrics.word_targets == 0U
                               ? 0.0F
                               : 100.0F * static_cast<float>(metrics.word_top1_hits) /
                                     static_cast<float>(metrics.word_targets);
    std::cout << label << " records=" << metrics.records
              << " batches=" << metrics.batches << " word_top1=" << std::fixed
              << std::setprecision(2) << accuracy << "% (" << metrics.word_top1_hits << '/'
              << metrics.word_targets << ")\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string training_path;
    std::string evaluation_path;
    std::string prompt = "the quick";
    TrainingOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--train" && index + 1 < argc) {
            training_path = argv[++index];
        } else if (argument == "--eval" && index + 1 < argc) {
            evaluation_path = argv[++index];
        } else if (argument == "--passes" && index + 1 < argc) {
            try {
                options.passes = std::stoull(argv[++index]);
                if (options.passes == 0U) {
                    throw std::invalid_argument("must be positive");
                }
            } catch (const std::exception& exception) {
                std::cerr << "Invalid --passes value: " << exception.what() << '\n';
                return 2;
            }
        } else if ((argument == "--batch-size" || argument == "--max-records" ||
                    argument == "--progress-every") &&
                   index + 1 < argc) {
            try {
                const std::size_t value = std::stoull(argv[++index]);
                if (argument == "--batch-size") {
                    if (value == 0U) {
                        throw std::invalid_argument("must be positive");
                    }
                    options.batch_size = value;
                } else if (argument == "--max-records") {
                    options.max_records = value;
                } else {
                    options.progress_every = value;
                }
            } catch (const std::exception& exception) {
                std::cerr << "Invalid " << argument << " value: " << exception.what() << '\n';
                return 2;
            }
        } else if (argument == "--prompt" && index + 1 < argc) {
            prompt = argv[++index];
        } else if (argument == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    try {
        HierarchicalTokenizer tokenizer = make_tokenizer();
        HierarchicalPredictionNetwork network = make_network();
        const PredictionMetrics training_metrics =
            training_path.empty() ? train_builtin(tokenizer, network, options)
                                  : train_file(tokenizer, network, training_path, options);
        const HierarchicalTokenTrace prompt_trace = tokenizer.encode(prompt);

        std::cout << "Online hierarchical training demo\n"
                  << "  training records: " << training_metrics.records << "\n"
                  << "  training batches: " << training_metrics.batches << "\n"
                  << "  batch size: " << options.batch_size << "\n"
                  << "  tokenizer words: " << tokenizer.words().size() << "\n"
                  << "  learned parts: " << tokenizer.parts().size() << "\n"
                  << "  network neurons: " << network.state().total_neuron_count
                  << " (tokenizer=" << network.state().tokenizer_neuron_count
                  << ", general=" << network.state().layer_neuron_count << ")\n"
                  << "  network steps: " << network.state().step << "\n"
                  << "  checkpointing: disabled\n";
        print_metrics("  online pre-update metrics:", training_metrics);

        if (!evaluation_path.empty()) {
            const PredictionMetrics evaluation_metrics =
                evaluate_file(tokenizer, network, evaluation_path, options.max_records);
            print_metrics("  held-out evaluation:", evaluation_metrics);
        }

        print_candidates("  first characters:",
                         network.predict_next_character(sequence_begin_id, 5U),
                         [](const PredictionCandidate& candidate) {
                             return character_label(candidate);
                         });
        print_candidates("  first words:",
                         network.predict_next_word(sequence_begin_id, 5U),
                         [&tokenizer](const PredictionCandidate& candidate) {
                             return word_label(candidate, tokenizer);
                         });

        if (!prompt_trace.atomic_tokens.empty()) {
            print_candidates("  next characters after prompt:",
                             network.predict_next_character(prompt_trace.atomic_tokens.back().id,
                                                            5U),
                             [](const PredictionCandidate& candidate) {
                                 return character_label(candidate);
                             });
        }
        if (!prompt_trace.words.empty()) {
            const auto& last_word = prompt_trace.words.back();
            if (last_word.word != agentari::text::hierarchical::WordVocabulary::unknown_id) {
                print_candidates("  next words after prompt:",
                                 network.predict_next_word(last_word.word, 5U),
                                 [&tokenizer](const PredictionCandidate& candidate) {
                                     return word_label(candidate, tokenizer);
                                 });
                print_candidates("  left-context predictions:",
                                 network.predict_left_context(last_word.word, 5U),
                                 [&tokenizer](const PredictionCandidate& candidate) {
                                     return word_label(candidate, tokenizer);
                                 });
            }
            for (const auto& span : last_word.parts.spans) {
                if (span.learned) {
                    const PartId part = span.id;
                    print_candidates("  next parts after prompt part:",
                                     network.predict_next_part(part, 5U),
                                     [&tokenizer](const PredictionCandidate& candidate) {
                                         return part_label(candidate, tokenizer);
                                     });
                    break;
                }
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "Online hierarchical demo failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
