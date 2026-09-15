#include "agentari/Tokenizer.hpp"

#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void layered_map_and_predictor() {
    using namespace agentari::text;

    LayeredTokenizer tokenizer(TokenizerConfig{
        .ascii_base_capacity = 256U,
        .ascii_expansion_capacity = 256U,
        .primitive_rule_capacity = 2U,
        .primitive_expansion_capacity = 2U,
        .vocabulary_capacity = 8U,
    });
    require(LayeredTokenizer::encode_ascii("A!") == std::vector<TokenId>{65U, 33U},
            "ASCII layer did not preserve bytes");
    std::string depth_error;
    require(tokenizer.load_depth_map_file("../data/token-depth-regions.tsv", depth_error),
            "token depth map did not load: " + depth_error);
    require(tokenizer.record(65U).depth == 224U &&
                std::fabs(tokenizer.token_mobility(65U) - 0.12F) < 1.0e-6F,
            "token depth map was not applied to ASCII learning mobility");
    require(tokenizer.record(256U).text == "<unused-ascii>", "ASCII expansion slot missing");
    require(tokenizer.add_primitive("ing") == 512U, "primitive layer id mismatch");
    require(tokenizer.add_primitive("bio") == 513U, "primitive layer id mismatch");
    require(tokenizer.add_primitive("un") == 514U, "primitive layer id mismatch");
    require(tokenizer.add_primitive("ness") == 515U, "primitive layer id mismatch");
    require(tokenizer.primitive_capacity() == 4U, "primitive capacity mismatch");
    require(tokenizer.add_word("the", 1U) == 517U, "word layer id mismatch");
    require(tokenizer.add_word("quick", 2U) == 518U, "word layer id mismatch");
    require(tokenizer.add_word("brown", 3U) == 519U, "word layer id mismatch");
    require(tokenizer.add_word("fox", 4U) == 520U, "word layer id mismatch");
    require(tokenizer.remaining_vocabulary_slots() == 4U, "word capacity accounting mismatch");
    require(tokenizer.word_token_count() == 4U, "dense word vocabulary size mismatch");
    require(tokenizer.model_vocabulary_size() == 5U, "unknown word class missing");
    require(tokenizer.word_index(520U) == 3U, "dense word index mismatch");
    require(tokenizer.token_from_word_index(3U) == 520U, "dense word reverse index mismatch");

    const auto mixed_text = tokenizer.encode_text("The, quick!\nunknown");
    require(tokenizer.decode_text(mixed_text) == "the, quick!\nunknown",
            "mixed tokenizer did not preserve separators and symbols");
    LayeredTokenizer fingerprint_tokenizer(tokenizer.config());
    const std::uint64_t empty_fingerprint = fingerprint_tokenizer.fingerprint();
    (void)fingerprint_tokenizer.add_word("fingerprint");
    require(fingerprint_tokenizer.fingerprint() != empty_fingerprint,
            "tokenizer fingerprint did not change after vocabulary growth");

    const auto encoded = tokenizer.encode_words("The quick brown unknown");
    require(encoded.size() == 4U, "word encoder size mismatch");
    require(encoded[0U] == tokenizer.lookup_word("the"), "word normalization mismatch");
    require(tokenizer.record(encoded.back()).text == "<unk>", "unknown word mismatch");

    const auto parts = tokenizer.encode_word_parts("unhappiness");
    require(parts.size() > 2U, "unknown word was not decomposed into fallback units");
    require(tokenizer.record(parts.front()).layer == TokenLayer::primitive,
            "unknown word did not use a primitive prefix");
    require(tokenizer.record(parts.back()).layer == TokenLayer::primitive,
            "unknown word did not use a primitive suffix");
    require(tokenizer.decode_word_parts(parts) == "unhappiness",
            "layered word fallback was not lossless");

    const PredictorConfig predictor_config{
        .maximum_context_words = 3U,
        .model_vocabulary_limit = 8U,
        .model_dimension = 32U,
        .layer_count = 1U,
        .attention_head_count = 4U,
        .key_value_head_count = 2U,
        .feed_forward_dimension = 64U,
        .maximum_sequence_length = 16U,
        .training_repeats_per_observation = 1U,
        .learning_rate = 1.0e-3F,
        .weight_decay = 0.0F,
        .seed = 1234U,
    };
    WordPredictor predictor(tokenizer, predictor_config);
    for (std::size_t step = 0U; step < 128U; ++step) {
        predictor.observe_text("the quick brown fox");
    }
    require(predictor.training_steps() == 128U, "neural predictor did not train");
    predictor.observe_text("the unknown @ symbol");
    const TokenizerNeuralState neural_state = predictor.tokenizer_state();
    require(neural_state.step > 128U, "tokenizer learning nodes did not receive input");
    const std::uint64_t symbol_flag = std::uint64_t{1U} <<
                                      static_cast<std::uint8_t>(PrimitiveFlag::symbol_input);
    const std::uint64_t unknown_flag = std::uint64_t{1U} <<
                                       static_cast<std::uint8_t>(PrimitiveFlag::unknown_word);
    require((neural_state.primitives.flags & symbol_flag) != 0U,
            "symbol input did not reach the tokenizer neuron stack");
    require((neural_state.primitives.flags & unknown_flag) != 0U,
            "unknown word did not reach the tokenizer neuron stack");
    require(neural_state.layers[0U].node_count == 256U,
            "ASCII learning-node capacity mismatch");
    require(neural_state.layers[1U].node_count == 1024U,
            "primitive learning-node capacity mismatch");
    require(neural_state.layers[2U].node_count == 4096U,
            "word learning-node capacity mismatch");
    require(neural_state.layers[3U].node_count == 2048U,
            "context learning-node capacity mismatch");
    require(neural_state.layers[4U].node_count == 256U,
            "grammar learning-node capacity mismatch");
    for (const LayerPrimitiveSummary& layer : neural_state.layers) {
        require(layer.region_activations.size() == tokenizer_region_count,
                "layer region summary size mismatch");
        require(layer.expert_activations.size() == 4U,
                "layer expert summary size mismatch");
    }
    const std::vector<float> message_feedback(tokenizer_feature_count, 0.8F);
    predictor.apply_message_feedback(message_feedback);
    predictor.observe_text("the");
    const TokenizerNeuralState feedback_state = predictor.tokenizer_state();
    const std::uint64_t message_flag = std::uint64_t{1U} <<
                                        static_cast<std::uint8_t>(PrimitiveFlag::message_feedback);
    require((feedback_state.primitives.flags & message_flag) != 0U,
            "message feedback did not reach the tokenizer neuron stack");
    const auto predictions = predictor.predict_text("the quick brown", 1U);
    require(predictions.size() == 1U && predictions.front().word == "fox",
            "word predictor did not learn the next word");
    require(predictor.generate("the quick brown", 1U) == "The quick brown fox.",
            "word generation mismatch");

    GrammarEngine grammar(tokenizer);
    grammar.add_exception("teh", "the");
    grammar.observe("the quick brown. another sentence");
    require(grammar.repair("  teh   quick ,brown") == "The quick, brown.",
            "grammar engine did not repair spacing, capitalization, and exceptions");
    require(grammar.state().learned_transitions > 0U,
            "grammar engine did not learn word transitions");

    PredictorConfig batched_config = predictor_config;
    batched_config.maximum_context_words = 4U;
    batched_config.maximum_sequence_length = 4U;
    batched_config.gradient_accumulation_steps = 2U;
    batched_config.maximum_training_windows_per_observation = 3U;
    batched_config.training_window_stride = 2U;
    WordPredictor batched_predictor(tokenizer, batched_config);
    batched_predictor.observe_text("the quick brown fox jumps over the lazy dog the quick brown fox");
    require(batched_predictor.training_steps() >= 2U,
            "batched predictor did not train multiple windows");
    require(std::isfinite(batched_predictor.evaluate_loss("the quick brown fox")),
            "batched predictor evaluation loss is not finite");

    const std::filesystem::path checkpoint =
        std::filesystem::temp_directory_path() / "agentari-tokenizer-test.ckpt";
    std::error_code remove_error;
    std::filesystem::remove(checkpoint, remove_error);
    std::string checkpoint_error;
    require(predictor.save_checkpoint(checkpoint.string(), checkpoint_error),
            "predictor checkpoint save failed: " + checkpoint_error);
    WordPredictor restored(tokenizer, predictor_config);
    require(restored.load_checkpoint(checkpoint.string(), checkpoint_error),
            "predictor checkpoint load failed: " + checkpoint_error);
    require(restored.training_steps() == predictor.training_steps(),
            "predictor checkpoint did not restore training steps");
    require(restored.predict_text("the quick brown", 1U).front().word == "fox",
            "predictor checkpoint did not restore learned output");
    std::filesystem::remove(checkpoint, remove_error);

    ContextSteering steering(tokenizer, ContextSteeringConfig{
        .local_window_words = 3U,
        .history_sample_count = 4U,
        .neighborhood_radius = 1U,
        .frame_word_count = 4U,
        .salient_words_per_frame = 2U,
        .retained_frame_count = 2U,
    });
    steering.record_chat_text(10U, "the quick brown fox");
    steering.record_chat_text(11U, "the quick brown fox");
    steering.set_goal_text(12U, "predict the next word");
    const ContextWindow context = steering.build_text(12U, "the quick brown");
    require(context.recent_words.size() == 3U, "local context window size mismatch");
    require(context.sampled_history_words.size() == 4U, "history sample size mismatch");
    require(context.frame_words.size() == 4U, "frame summary size mismatch");
    require(context.goal_words.size() == 4U, "goal context size mismatch");
    require(!context.flatten().empty(), "flattened context is empty");
    const auto limited_context = context.flatten(3U);
    require(limited_context.size() == 3U && limited_context.back() == context.recent_words.back(),
            "bounded context did not preserve the live suffix");
    require(steering.frame_count() == 2U, "retained frame count mismatch");
    require(predictor.predict(context, 1U).front().word == "fox",
            "context steering did not preserve local prediction suffix");
}

}  // namespace

int main() {
    try {
        layered_map_and_predictor();
        std::cout << "agentari tokenizer tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << "agentari tokenizer tests failed: " << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}
