#include "firstagent/Hardware.hpp"
#include "firstagent/Tokenizer.hpp"
#include "firstagent/VulkanBackend.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <string>
#include <string_view>

namespace {

std::filesystem::path project_file(const std::filesystem::path& relative) {
#ifdef FIRSTAGENT_PROJECT_SOURCE_DIR
    const std::filesystem::path source_path =
        std::filesystem::path(FIRSTAGENT_PROJECT_SOURCE_DIR) / relative;
    if (std::filesystem::exists(source_path)) {
        return source_path;
    }
#endif
    return relative;
}

void print_help() {
    std::cout << "Commands:\n"
              << "  /learn TEXT       train one sequence\n"
              << "  /predict TEXT     show the top next words\n"
              << "  /generate TEXT    generate four words\n"
              << "  /goal TEXT        set the current goal context\n"
              << "  /history TEXT     add a chat log and move to the next log\n"
              << "  /feedback VALUE   apply external message feedback (0..1)\n"
              << "  /save PATH        save model, optimizer, and tokenizer learning state\n"
              << "  /load PATH        restore a compatible checkpoint\n"
              << "  /gpu              run the Vulkan matrix self-test\n"
              << "  /state            inspect tokenizer neuron summaries\n"
              << "  /help             show this help\n"
              << "  /quit             exit\n"
              << "\nPlain text is treated as a prediction prompt.\n";
}

void print_state(const firstagent::text::TokenizerNeuralState& state) {
    std::cout << "tokenizer step=" << state.step << " flags=0x" << std::hex
              << state.primitives.flags << std::dec << '\n';
    constexpr const char* layer_names[] = {"ascii", "primitive", "word", "context", "grammar"};
    for (std::size_t index = 0U; index < state.layers.size(); ++index) {
        const auto& layer = state.layers[index];
        std::cout << "  " << layer_names[index] << ": total=" << layer.total_activation
                  << " center=" << layer.center_of_mass
                  << " novelty=" << layer.novelty
                  << " uncertainty=" << layer.uncertainty
                  << " up=" << layer.upward_signal
                  << " down=" << layer.downward_feedback
                  << " expert=" << layer.active_expert << '\n';
    }
    std::cout << "  primitive values:";
    for (const float value : state.primitives.values) {
        std::cout << ' ' << std::fixed << std::setprecision(3) << value;
    }
    std::cout << std::defaultfloat << "\n  regions:";
    for (const std::uint8_t value : state.primitives.region_states) {
        std::cout << ' ' << static_cast<unsigned int>(value);
    }
    std::cout << "\n  feedback: recurrent=" << state.recurrent_feedback[0U]
              << " message=" << state.message_feedback[0U] << '\n';
}

void print_predictions(const std::vector<firstagent::text::Prediction>& predictions) {
    if (predictions.empty()) {
        std::cout << "No prediction yet. Use /learn a few times first.\n";
        return;
    }
    std::cout << "next:";
    for (const auto& prediction : predictions) {
        std::cout << ' ' << prediction.word << " (" << prediction.probability << ')';
    }
    std::cout << '\n';
}

bool gpu_self_test(firstagent::gpu::VulkanComputeBackend& gpu) {
    if (!gpu.available()) {
        std::cout << "GPU backend unavailable: " << gpu.error() << '\n';
        return false;
    }
    const float left[] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    const float right[] = {7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    float output[4]{};
    if (!gpu.matmul(left, 2U, 3U, right, 2U, output)) {
        std::cout << "GPU matrix dispatch failed.\n";
        return false;
    }
    const float expected[] = {58.0F, 64.0F, 139.0F, 154.0F};
    const bool correct = std::equal(std::begin(output), std::end(output), std::begin(expected),
                                   [](float actual, float wanted) {
                                       return std::abs(actual - wanted) < 0.001F;
                                   });
    std::cout << "GPU: " << gpu.device_name() << " | matmul self-test: "
              << (correct ? "passed" : "FAILED") << '\n';
    return correct;
}

}  // namespace

int main(int argc, char* argv[]) {
    using firstagent::text::ContextSteering;
    using firstagent::text::LayeredTokenizer;
    using firstagent::text::PredictorConfig;
    using firstagent::text::WordPredictor;

    LayeredTokenizer tokenizer;
    std::string error;
    (void)tokenizer.load_depth_map_file(project_file("data/token-depth-regions.tsv").string(), error);
    const auto primitive_path = project_file("data/word-building-rules.tsv");
    if (!tokenizer.load_primitive_file(primitive_path.string(), 2048U, error)) {
        for (const std::string primitive : {"aero", "bio", "chem", "data", "geo", "mech", "neuro"}) {
            (void)tokenizer.add_primitive(primitive);
        }
    }
    const std::filesystem::path vocabulary_path =
        "/var/mnt/Storage/AI-Data/FirstAgent/Tokenizer/wordfreq-en-250000/"
        "wordfreq-en-common-250000.tsv";
    if (std::filesystem::exists(vocabulary_path)) {
        (void)tokenizer.load_vocabulary_file(vocabulary_path.string(), 100000U, error);
    }
    for (const std::string word : {"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                                   "predict", "next", "meaningful", "word"}) {
        (void)tokenizer.add_word(word);
    }

    WordPredictor predictor(tokenizer, PredictorConfig{
        .maximum_context_words = 512U,
        .model_vocabulary_limit = 2048U,
        .model_dimension = 32U,
        .layer_count = 1U,
        .attention_head_count = 4U,
        .key_value_head_count = 2U,
        .feed_forward_dimension = 128U,
        .maximum_sequence_length = 512U,
        .training_repeats_per_observation = 1U,
        .learning_rate = 1.0e-3F,
        .weight_decay = 0.0F,
        .seed = 0x70'70'0001ULL,
    });
    for (std::size_t step = 0U; step < 8U; ++step) {
        predictor.observe_text("the quick brown fox jumps over the lazy dog");
    }

    ContextSteering steering(tokenizer);
    steering.record_chat_text(1U, "the quick brown fox jumps over the lazy dog");
    std::uint64_t current_log = 2U;
    std::string current_prompt = "the quick brown";
    std::filesystem::path executable_path = argc > 0 ? std::filesystem::absolute(argv[0]) : std::filesystem::path{};
    std::filesystem::path shader_path = executable_path.parent_path() / "firstagent-matmul.comp.spv";
    firstagent::gpu::VulkanComputeBackend gpu(shader_path.string());

    std::cout << "FirstAgent toy\n"
              << "CPU SIMD: " << firstagent::hardware::preferred_simd_backend() << "\n"
              << "Tokenizer words: " << tokenizer.vocabulary_size()
              << " | active model words: " << predictor.active_vocabulary_size() << "\n";
    (void)gpu_self_test(gpu);
    print_help();

    std::string line;
    while (std::cout << "> " && std::getline(std::cin, line)) {
        if (line == "/quit" || line == "/exit") {
            break;
        }
        if (line == "/help") {
            print_help();
            continue;
        }
        if (line == "/gpu") {
            (void)gpu_self_test(gpu);
            continue;
        }
        if (line == "/state") {
            print_state(predictor.tokenizer_state());
            continue;
        }
        const auto argument = [&line](std::string_view command) {
            return std::string_view(line).substr(command.size());
        };
        if (line.rfind("/learn ", 0U) == 0U) {
            predictor.observe_text(argument("/learn "));
            std::cout << "trained step " << predictor.training_steps()
                      << ", loss " << predictor.last_loss() << '\n';
        } else if (line.rfind("/feedback ", 0U) == 0U) {
            float value = 0.0F;
            try {
                value = std::stof(std::string(argument("/feedback ")));
            } catch (const std::exception&) {
                std::cout << "Feedback must be a number from 0 to 1.\n";
                continue;
            }
            if (!(value >= 0.0F && value <= 1.0F)) {
                std::cout << "Feedback must be a number from 0 to 1.\n";
                continue;
            }
            predictor.apply_message_feedback(std::vector<float>(firstagent::text::tokenizer_feature_count,
                                                                value));
            std::cout << "message feedback applied\n";
        } else if (line.rfind("/save ", 0U) == 0U) {
            const std::string path(argument("/save "));
            if (path.empty() || !predictor.save_checkpoint(path, error)) {
                std::cout << "Checkpoint save failed: " << (path.empty() ? "missing path" : error) << '\n';
            } else {
                std::cout << "checkpoint saved\n";
            }
        } else if (line.rfind("/load ", 0U) == 0U) {
            const std::string path(argument("/load "));
            if (path.empty() || !predictor.load_checkpoint(path, error)) {
                std::cout << "Checkpoint load failed: " << (path.empty() ? "missing path" : error) << '\n';
            } else {
                std::cout << "checkpoint loaded\n";
            }
        } else if (line.rfind("/goal ", 0U) == 0U) {
            steering.set_goal_text(current_log, argument("/goal "));
            std::cout << "goal stored for log " << current_log << '\n';
        } else if (line.rfind("/history ", 0U) == 0U) {
            current_prompt = std::string(argument("/history "));
            steering.record_chat_text(current_log, current_prompt);
            ++current_log;
            std::cout << "chat log stored; current log is " << current_log << '\n';
        } else if (line.rfind("/predict ", 0U) == 0U) {
            current_prompt = std::string(argument("/predict "));
            print_predictions(predictor.predict(steering.build_text(current_log, current_prompt), 5U));
        } else if (line.rfind("/generate ", 0U) == 0U) {
            current_prompt = std::string(argument("/generate "));
            std::cout << predictor.generate(current_prompt, 4U, 1U) << '\n';
        } else if (!line.empty()) {
            current_prompt = line;
            print_predictions(predictor.predict(steering.build_text(current_log, current_prompt), 5U));
        }
    }
    return 0;
}
