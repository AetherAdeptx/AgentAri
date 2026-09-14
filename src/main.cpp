#include "firstagent/Agent.hpp"
#include "firstagent/AgentWorker.hpp"
#include "firstagent/NeuralNetwork.hpp"
#include "firstagent/Parallel.hpp"
#include "firstagent/SdlWindow.hpp"
#include "firstagent/System.hpp"
#include "firstagent/Tokenizer.hpp"
#include "firstagent/VulkanBackend.hpp"

#include <SDL3/SDL_main.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string local_time(const std::vector<std::string>&) {
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local{};
    if (const auto* value = std::localtime(&now); value != nullptr) {
        local = *value;
    }

    std::ostringstream output;
    output << std::put_time(&local, "%Y-%m-%d %H:%M:%S %Z");
    return output.str();
}

std::string repeat_text(const std::vector<std::string>& arguments) {
    if (arguments.empty()) {
        return "repeat requires text";
    }

    std::ostringstream output;
    for (std::size_t index = 0; index < arguments.size(); ++index) {
        if (index != 0U) {
            output << ' ';
        }
        output << arguments[index];
    }
    return output.str();
}

std::filesystem::path project_file(const std::filesystem::path& relative) {
#ifdef FIRSTAGENT_PROJECT_SOURCE_DIR
    const std::filesystem::path source = std::filesystem::path(FIRSTAGENT_PROJECT_SOURCE_DIR) / relative;
    if (std::filesystem::exists(source)) {
        return source;
    }
#endif
    return relative;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [--memory PATH]\n"
              << "       " << program << " [--memory PATH] [--run-for-ms N]\n"
              << "       " << program << " [--workers N] [--vulkan]\n"
              << "       " << program << " --nn-demo\n"
              << "       " << program << " --word-demo\n"
              << "\n--workers 0 selects usable logical CPUs minus two; a positive value overrides it.\n"
              << "\nThe AI scheduler runs at a fixed 8 ms interval (125 Hz).\n";
}

int run_word_demo() {
    using firstagent::text::LayeredTokenizer;
    using firstagent::text::WordPredictor;

    LayeredTokenizer tokenizer;
    std::string load_error;
    (void)tokenizer.load_depth_map_file(project_file("data/token-depth-regions.tsv").string(), load_error);
    if (!tokenizer.load_primitive_file(project_file("data/word-building-rules.tsv").string(),
                                       2048U, load_error)) {
        for (const std::string primitive : {"aero", "astro", "bio", "chem", "cosmo", "data", "geo", "mech", "neuro", "system"}) {
            (void)tokenizer.add_primitive(primitive);
        }
    }

    const std::filesystem::path ranked_vocabulary =
        "/var/mnt/Storage/AI-Data/FirstAgent/Tokenizer/wordfreq-en-250000/"
        "wordfreq-en-common-250000.tsv";
    if (std::filesystem::exists(ranked_vocabulary)) {
        if (!tokenizer.load_vocabulary_file(ranked_vocabulary.string(), 100000U, load_error)) {
            std::cerr << "Vocabulary load warning: " << load_error << '\n';
        }
    } else {
        for (const std::string word : {"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"}) {
            (void)tokenizer.add_word(word);
        }
    }

    WordPredictor predictor(tokenizer);
    predictor.observe_text("the quick brown fox jumps over the lazy dog");
    predictor.observe_text("the quick brown fox jumps over the lazy dog");

    const auto predictions = predictor.predict_text("the quick brown", 3U);
    std::cout << "Word predictor: ASCII=256+256 growth, primitives=" << tokenizer.primitive_count()
              << ", words=" << tokenizer.vocabulary_size() << "/"
              << tokenizer.vocabulary_capacity() << ", training-steps=" << predictor.training_steps()
              << "\n  next:";
    for (const auto& prediction : predictions) {
        std::cout << ' ' << prediction.word << " (" << prediction.probability << ')';
    }
    std::cout << "\n  generated: " << predictor.generate("the quick brown", 4U) << '\n';
    return 0;
}

int run_neural_demo() {
    using firstagent::nn::AdamW;
    using firstagent::nn::GenerationConfig;
    using firstagent::nn::TransformerConfig;
    using firstagent::nn::TransformerModel;

    TransformerConfig config;
    config.vocabulary_size = 16U;
    config.maximum_sequence_length = 32U;
    config.model_dimension = 32U;
    config.layer_count = 2U;
    config.attention_head_count = 4U;
    config.key_value_head_count = 2U;
    config.feed_forward_dimension = 64U;
    config.seed = 20260913ULL;

    TransformerModel model(config);
    AdamW optimizer(model.parameters(), 2.0e-3F);
    const std::vector<std::size_t> input{1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U};
    const std::vector<std::size_t> target{2U, 3U, 4U, 5U, 6U, 7U, 8U, 9U};

    std::cout << "Neural demo: " << model.parameter_count() << " parameters, "
              << config.layer_count << " Transformer layers, "
              << config.attention_head_count << " query heads, "
              << config.key_value_head_count << " key/value heads.\n";
    for (std::size_t update = 0U; update < 12U; ++update) {
        const auto loss = model.loss(input, target);
        loss.backward();
        firstagent::nn::clip_grad_norm(model.parameters(), 1.0F);
        optimizer.step();
        optimizer.zero_grad();
        if (update == 0U || update == 11U) {
            std::cout << "  update " << (update + 1U) << " loss=" << loss.values().front() << '\n';
        }
    }

    const auto generated = model.generate({1U, 2U, 3U}, GenerationConfig{
        .maximum_new_tokens = 5U,
        .temperature = 0.0F,
        .top_k = 0U,
        .seed = 7U,
    });
    std::cout << "  cached generation:";
    for (const std::size_t token : generated) {
        std::cout << ' ' << token;
    }
    std::cout << '\n';
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::filesystem::path memory_path = "data/experiences.tsv";
    std::chrono::milliseconds run_for{0};
    bool neural_demo = false;
    bool word_demo = false;
    bool vulkan_requested = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--memory" && index + 1 < argc) {
            memory_path = argv[++index];
        } else if (argument == "--run-for-ms" && index + 1 < argc) {
            try {
                const auto milliseconds = std::stoll(argv[++index]);
                if (milliseconds <= 0) {
                    throw std::invalid_argument("must be positive");
                }
                run_for = std::chrono::milliseconds(milliseconds);
            } catch (const std::exception& exception) {
                std::cerr << "Invalid --run-for-ms value: " << exception.what() << '\n';
                return 2;
            }
        } else if (argument == "--workers" && index + 1 < argc) {
            try {
                const std::string worker_argument = argv[++index];
                if (!worker_argument.empty() && worker_argument.front() == '-') {
                    throw std::invalid_argument("must be zero or positive");
                }
                const auto workers = std::stoull(worker_argument);
                firstagent::parallel::set_default_worker_count(
                    static_cast<std::size_t>(workers));
            } catch (const std::exception& exception) {
                std::cerr << "Invalid --workers value: " << exception.what() << '\n';
                return 2;
            }
        } else if (argument == "--nn-demo") {
            neural_demo = true;
        } else if (argument == "--word-demo") {
            word_demo = true;
        } else if (argument == "--vulkan") {
            vulkan_requested = true;
        } else if (argument == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argument << '\n';
            print_usage(argv[0]);
            return 2;
        }
    }

    if (neural_demo) {
        return run_neural_demo();
    }
    if (word_demo) {
        return run_word_demo();
    }

    std::optional<firstagent::gpu::VulkanComputeBackend> gpu_backend;
    if (vulkan_requested) {
        gpu_backend.emplace();
        if (gpu_backend->available()) {
            firstagent::nn::set_tensor_backend(firstagent::nn::TensorBackend::vulkan,
                                               &*gpu_backend);
            std::cout << "Tensor backend: Vulkan (" << gpu_backend->device_name() << ")\n";
        } else {
            std::cerr << "Vulkan requested but unavailable: " << gpu_backend->error()
                      << "; using CPU tensors.\n";
        }
    }

    firstagent::Memory memory(memory_path);
    std::string error;
    if (!memory.load(error)) {
        std::cerr << "Failed to load memory: " << error << '\n';
        return 1;
    }

    firstagent::ToolRegistry tools;
    if (!tools.register_tool({"time", "Return the local system time.", local_time}, error) ||
        !tools.register_tool({"repeat", "Return the supplied text.", repeat_text}, error)) {
        std::cerr << "Failed to register tool: " << error << '\n';
        return 1;
    }

    firstagent::text::LayeredTokenizer tokenizer;
    (void)tokenizer.load_depth_map_file(project_file("data/token-depth-regions.tsv").string(), error);
    if (!tokenizer.load_primitive_file(project_file("data/word-building-rules.tsv").string(),
                                       2048U, error)) {
        for (const std::string primitive : {"aero", "astro", "bio", "chem", "cosmo",
                                             "data", "geo", "mech", "neuro", "system"}) {
            (void)tokenizer.add_primitive(primitive);
        }
    }
    const std::filesystem::path ranked_vocabulary =
        "/var/mnt/Storage/AI-Data/FirstAgent/Tokenizer/wordfreq-en-250000/"
        "wordfreq-en-common-250000.tsv";
    if (std::filesystem::exists(ranked_vocabulary)) {
        if (!tokenizer.load_vocabulary_file(ranked_vocabulary.string(), 100000U, error)) {
            std::cerr << "Vocabulary load warning: " << error << '\n';
        }
    } else {
        for (const std::string word : {"the", "quick", "brown", "fox", "jumps", "over",
                                       "the", "lazy", "dog"}) {
            (void)tokenizer.add_word(word);
        }
    }
    firstagent::text::PredictorConfig predictor_config;
    predictor_config.maximum_context_words = 256U;
    predictor_config.maximum_sequence_length = 256U;
    predictor_config.model_vocabulary_limit = 8192U;
    firstagent::text::WordPredictor predictor(tokenizer, predictor_config);
    firstagent::text::ContextSteering context(tokenizer, firstagent::text::ContextSteeringConfig{
        .local_window_words = 256U,
        .history_sample_count = 256U,
        .neighborhood_radius = 4U,
        .frame_word_count = 4096U,
        .salient_words_per_frame = 8U,
        .retained_frame_count = 32U,
    });
    firstagent::Agent agent(memory, tools, &predictor, &context);
    firstagent::AgentWorker agent_worker(agent);
    firstagent::SdlWindow window;
    if (!window.open(error)) {
        std::cerr << "Failed to open runtime window: " << error << '\n';
        return 1;
    }

    using Clock = std::chrono::steady_clock;
    constexpr auto ai_period = std::chrono::milliseconds(8);
    constexpr auto render_period = std::chrono::milliseconds(16);
    const auto start = Clock::now();
    auto next_ai_tick = start;
    auto next_render = start;
    std::deque<std::string> pending_commands;
    const firstagent::system::CpuCapabilities capabilities =
        firstagent::system::cpu_capabilities();
    const firstagent::system::GpuCapabilities gpu_capabilities =
        firstagent::system::gpu_capabilities();
    std::string status = "Learning worker ready (" +
                         std::string(firstagent::system::simd_name(capabilities.simd)) +
                         ", " + std::to_string(capabilities.topology.usable_processors) +
                         " usable CPU threads, " +
                         std::to_string(firstagent::parallel::default_worker_count()) +
                         " parallel workers, " +
                         std::to_string(gpu_capabilities.device_count) + " DRM GPU(s)).";
    bool input_closed = false;
    bool running = true;
    std::uint64_t ai_ticks = 0;
    std::uint64_t rendered_frames = 0;
    constexpr std::size_t maximum_ai_catch_up_ticks = 4U;

    while (running) {
        bool window_closed = false;
        if (!window.pump_events(pending_commands, window_closed, error)) {
            status = "Input error: " + error;
            window_closed = true;
        }
        input_closed = input_closed || window_closed;

        while (const auto result = agent_worker.poll()) {
            status = result->response;
        }

        const auto now = Clock::now();
        std::size_t catch_up_ticks = 0U;
        while (now >= next_ai_tick && running && catch_up_ticks < maximum_ai_catch_up_ticks) {
            ++ai_ticks;
            if (!pending_commands.empty()) {
                const std::string command = std::move(pending_commands.front());
                pending_commands.pop_front();
                if (command == "/quit") {
                    running = false;
                } else if (!agent_worker.submit(command)) {
                    status = "Agent queue is full; input was deferred.";
                }
            }
            next_ai_tick += ai_period;
            ++catch_up_ticks;
        }
        if (now >= next_ai_tick) {
            next_ai_tick = now + ai_period;
            status = "AI catch-up capped after a delayed frame.";
        }

        const auto after_ai = Clock::now();
        if (after_ai >= next_render) {
            ++rendered_frames;
            window.render(firstagent::WindowState{
                .uptime = std::chrono::duration_cast<std::chrono::milliseconds>(after_ai - start),
                .ai_ticks = ai_ticks,
                .rendered_frames = rendered_frames,
                .input = window.input_buffer(),
                .status = status,
            });
            next_render += render_period;
        }

        if (run_for.count() > 0 && after_ai - start >= run_for) {
            running = false;
            continue;
        }

        if (input_closed && pending_commands.empty()) {
            running = false;
            continue;
        }

        const auto deadline = std::min(next_ai_tick, next_render);
        const auto current = Clock::now();
        if (deadline > current) {
            window.wait_for_next_tick(
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - current));
        }
    }

    window.close();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start);
    std::cout << "FirstAgent stopped. AI ticks: " << ai_ticks
              << ", rendered frames: " << rendered_frames
              << ", elapsed: " << elapsed.count() << " ms"
              << ", experiences stored: " << memory.size() << '\n';
    return 0;
}
