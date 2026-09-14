#include "firstagent/AgentWorker.hpp"
#include "firstagent/Tokenizer.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    const std::filesystem::path memory_path =
        std::filesystem::temp_directory_path() / "firstagent-agent-worker-test.tsv";
    std::error_code filesystem_error;
    std::filesystem::remove(memory_path, filesystem_error);
    try {
        firstagent::Memory memory(memory_path);
        std::string error;
        require(memory.load(error), "worker test memory failed to load: " + error);
        firstagent::ToolRegistry tools;
        firstagent::text::LayeredTokenizer tokenizer(firstagent::text::TokenizerConfig{
            .ascii_base_capacity = 256U,
            .ascii_expansion_capacity = 256U,
            .primitive_rule_capacity = 2U,
            .primitive_expansion_capacity = 2U,
            .vocabulary_capacity = 8U,
        });
        (void)tokenizer.add_word("the", 1U);
        (void)tokenizer.add_word("quick", 2U);
        (void)tokenizer.add_word("brown", 3U);
        (void)tokenizer.add_word("fox", 4U);
        firstagent::text::PredictorConfig predictor_config;
        predictor_config.maximum_context_words = 3U;
        predictor_config.model_vocabulary_limit = 8U;
        predictor_config.model_dimension = 16U;
        predictor_config.layer_count = 1U;
        predictor_config.attention_head_count = 4U;
        predictor_config.key_value_head_count = 2U;
        predictor_config.feed_forward_dimension = 32U;
        predictor_config.maximum_sequence_length = 16U;
        predictor_config.weight_decay = 0.0F;
        firstagent::text::WordPredictor predictor(tokenizer, predictor_config);
        firstagent::text::ContextSteering context(tokenizer,
                                                   firstagent::text::ContextSteeringConfig{
                                                       .local_window_words = 3U,
                                                       .history_sample_count = 4U,
                                                       .neighborhood_radius = 1U,
                                                       .frame_word_count = 4U,
                                                       .salient_words_per_frame = 2U,
                                                       .retained_frame_count = 2U,
                                                   });
        firstagent::Agent agent(memory, tools, &predictor, &context);
        firstagent::AgentWorker worker(agent, 2U);
        require(worker.submit("/remember background fact"), "worker rejected first request");

        std::optional<firstagent::AgentWorker::Result> result;
        for (std::size_t attempt = 0U; attempt < 200U && !result; ++attempt) {
            result = worker.poll();
            if (!result) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        require(result.has_value(), "worker did not return a response");
        require(result->response.find("Stored experience") != std::string::npos,
                "worker response did not come from the agent");
        require(memory.size() == 1U, "background agent did not persist memory");
        require(result->sequence == 0U, "worker sequence did not start at zero");
        require(worker.submit("the quick brown"), "worker rejected predictor request");
        result.reset();
        for (std::size_t attempt = 0U; attempt < 300U && !result; ++attempt) {
            result = worker.poll();
            if (!result) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
        require(result.has_value(), "worker did not return predictor response");
        require(result->response.find("I heard: the quick brown") != std::string::npos,
                "predictor worker response lost the original input");
        require(memory.size() == 2U, "predictor worker did not persist observation");
        std::filesystem::remove(memory_path, filesystem_error);
        std::cout << "firstagent agent worker tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::filesystem::remove(memory_path, filesystem_error);
        std::cerr << "firstagent agent worker tests failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
