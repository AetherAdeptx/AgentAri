#include "agentari/Tokenizer.hpp"
#include "agentari/Hardware.hpp"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [--train PATH] [--max-bytes N] [--prompt TEXT]\n";
}

std::filesystem::path project_file(const std::filesystem::path& relative) {
#ifdef AGENTARI_PROJECT_SOURCE_DIR
    const std::filesystem::path source = std::filesystem::path(AGENTARI_PROJECT_SOURCE_DIR) / relative;
    if (std::filesystem::exists(source)) {
        return source;
    }
#endif
    return relative;
}

}  // namespace

int main(int argc, char* argv[]) {
    using agentari::text::LayeredTokenizer;
    using agentari::text::ContextSteering;
    using agentari::text::WordPredictor;

    std::filesystem::path training_path;
    std::string prompt = "the quick brown";
    std::size_t maximum_bytes = 0U;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--train" && index + 1 < argc) {
            training_path = argv[++index];
        } else if (argument == "--max-bytes" && index + 1 < argc) {
            try {
                maximum_bytes = std::stoull(argv[++index]);
            } catch (const std::exception&) {
                std::cerr << "--max-bytes must be a non-negative integer\n";
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

    LayeredTokenizer tokenizer;
    std::string error;
    (void)tokenizer.load_depth_map_file(project_file("data/token-depth-regions.tsv").string(), error);
    if (!tokenizer.load_primitive_file(project_file("data/word-building-rules.tsv").string(), 2048U, error)) {
        std::cerr << "Primitive load warning: " << error << '\n';
        for (const std::string primitive : {"aero", "astro", "bio", "chem", "cosmo", "data", "geo", "mech", "neuro", "system"}) {
            (void)tokenizer.add_primitive(primitive);
        }
    }

    const std::filesystem::path ranked_vocabulary =
        "/var/mnt/Storage/AI-Data/AgentAri/Tokenizer/wordfreq-en-250000/"
        "wordfreq-en-common-250000.tsv";
    if (std::filesystem::exists(ranked_vocabulary) &&
        !tokenizer.load_vocabulary_file(ranked_vocabulary.string(), 100000U, error)) {
        std::cerr << "Vocabulary load warning: " << error << '\n';
    }
    for (const std::string word : {"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"}) {
        (void)tokenizer.add_word(word);
    }

    WordPredictor predictor(tokenizer);
    if (!training_path.empty()) {
        if (!predictor.observe_file(training_path.string(), maximum_bytes, error)) {
            std::cerr << "Training failed: " << error << '\n';
            return 1;
        }
    } else {
        for (std::size_t step = 0U; step < 32U; ++step) {
            predictor.observe_text("the quick brown fox jumps over the lazy dog");
        }
    }

    ContextSteering steering(tokenizer);
    steering.record_chat_text(1U, "the quick brown fox jumps over the lazy dog");
    steering.set_goal_text(2U, "predict the next meaningful word");
    const auto context = steering.build_text(2U, prompt);
    const auto predictions = predictor.predict(context, 3U);
    std::cout << "Word predictor: ASCII=256+256 growth, primitives=" << tokenizer.primitive_count()
              << ", words=" << tokenizer.vocabulary_size() << "/"
              << tokenizer.vocabulary_capacity() << ", training-steps=" << predictor.training_steps()
              << ", active-model-words=" << predictor.active_vocabulary_size()
              << ", loss=" << predictor.last_loss()
              << ", simd=" << agentari::hardware::preferred_simd_backend()
              << ", local=" << context.recent_words.size()
              << ", sampled-history=" << context.sampled_history_words.size()
              << ", frame-words=" << context.frame_words.size()
              << ", goal=" << context.goal_words.size()
              << "\n  next:";
    for (const auto& prediction : predictions) {
        std::cout << ' ' << prediction.word << " (" << prediction.probability << ')';
    }
    std::cout << "\n  generated: " << predictor.generate(prompt, 4U) << '\n';
    return 0;
}
