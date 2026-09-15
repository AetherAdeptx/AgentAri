#include "agentari/Agent.hpp"
#include "agentari/AgentWorker.hpp"
#include "agentari/HierarchicalPrediction.hpp"
#include "agentari/NeuralNetwork.hpp"
#include "agentari/Parallel.hpp"
#include "agentari/RunLog.hpp"
#include "agentari/SdlWindow.hpp"
#include "agentari/System.hpp"
#include "agentari/Tokenizer.hpp"
#include "agentari/VulkanBackend.hpp"

#include <SDL3/SDL_main.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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
#ifdef AGENTARI_PROJECT_SOURCE_DIR
    const std::filesystem::path source = std::filesystem::path(AGENTARI_PROJECT_SOURCE_DIR) / relative;
    if (std::filesystem::exists(source)) {
        return source;
    }
#endif
    return relative;
}

struct HierarchicalRuntime {
    agentari::text::hierarchical::HierarchicalTokenizer tokenizer;
    agentari::text::hierarchical::HierarchicalPredictionNetwork network;
};

agentari::text::hierarchical::PredictionNetworkConfig hierarchical_network_config(
    const std::size_t total_neurons,
    const std::size_t tokenizer_neurons) {
    using namespace agentari::text::hierarchical;
    PredictionNetworkConfig config;
    config.total_neuron_count = total_neurons;
    config.tokenizer_neuron_count = tokenizer_neurons;
    config.neurons_per_layer = 2000U;
    config.additional_layer_count = 24U;
    config.min_neurons_per_layer = Min_Neurons_Per_Layer;
    return config;
}

HierarchicalRuntime start_hierarchical_runtime(const std::size_t total_neurons,
                                               const std::size_t tokenizer_neurons) {
    using namespace agentari::text::hierarchical;

    HierarchicalRuntime runtime{
        .tokenizer = HierarchicalTokenizer(HierarchicalTokenizerConfig{
            .parts = SemanticPartLearnerConfig{
                .capacity = 1024U,
                .minimum_part_length = 2U,
                .maximum_part_length = 16U,
                .minimum_distinct_words = 2U,
                .context_radius = 2U,
            },
            .vocabulary_capacity = 100000U,
            .context = ContextTokenizerConfig{.recent_word_limit = 256U},
        }),
        .network = HierarchicalPredictionNetwork(
            hierarchical_network_config(total_neurons, tokenizer_neurons)),
    };
    return runtime;
}

class RuntimeTestWorkspace {
public:
    explicit RuntimeTestWorkspace(const bool enabled) {
        if (!enabled) {
            return;
        }
        enabled_ = true;

        std::error_code filesystem_error;
        root_ = std::filesystem::temp_directory_path(filesystem_error);
        if (filesystem_error) {
            error_ = "could not find the temporary directory: " + filesystem_error.message();
            root_.clear();
            return;
        }
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root_ /= "agentari-runtime-test-" + std::to_string(stamp);
        std::filesystem::create_directories(root_, filesystem_error);
        if (filesystem_error) {
            error_ = "could not create the temporary test workspace: " +
                     filesystem_error.message();
            root_.clear();
            return;
        }
        owns_root_ = true;
    }

    ~RuntimeTestWorkspace() {
        if (owns_root_) {
            std::error_code filesystem_error;
            std::filesystem::remove_all(root_, filesystem_error);
        }
    }

    RuntimeTestWorkspace(const RuntimeTestWorkspace&) = delete;
    RuntimeTestWorkspace& operator=(const RuntimeTestWorkspace&) = delete;

    [[nodiscard]] bool ready() const noexcept {
        return !enabled_ || (error_.empty() && !root_.empty());
    }

    [[nodiscard]] const std::string& error() const noexcept {
        return error_;
    }

    [[nodiscard]] const std::filesystem::path& root() const noexcept {
        return root_;
    }

    [[nodiscard]] std::filesystem::path file(const std::filesystem::path& relative) const {
        return root_ / relative;
    }

private:
    std::filesystem::path root_;
    std::string error_;
    bool enabled_{false};
    bool owns_root_{false};
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [--memory PATH]\n"
              << "       " << program << " [--memory PATH] [--run-for-ms N]\n"
              << "       " << program << " --test [--run-for-ms N]\n"
              << "       " << program << " [--total-neurons N] [--tokenizer-neurons N]\n"
              << "               [--workers N] [--log PATH]\n"
              << "               [--max-cpu-usage PCT] [--max-gpu-usage PCT] [--no-gui]\n"
              << "               [--vulkan]\n"
              << "       " << program << " --nn-demo\n"
              << "       " << program << " --word-demo\n"
              << "\n--test uses a temporary runtime workspace and removes it on shutdown.\n"
              << "\n--workers 0 selects usable logical CPUs minus two; a positive value overrides it.\n"
              << "--total-neurons N sets the hierarchical network budget (default: 56,000).\n"
              << "--tokenizer-neurons N fixes the aggregate tokenizer pool (default: 8,000).\n"
              << "--max-cpu-usage PCT caps the scheduler's worker-thread budget.\n"
              << "--max-gpu-usage PCT records the future GPU scheduler budget.\n"
              << "--no-gui runs the 8 ms AI loop without opening SDL; pair with --run-for-ms.\n"
              << "--log PATH removes the previous log at that exact path and starts a fresh transient log.\n"
              << "\nThe AI scheduler runs at a fixed 8 ms interval (125 Hz).\n";
}

int run_word_demo() {
    using agentari::text::LayeredTokenizer;
    using agentari::text::WordPredictor;

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
        "/var/mnt/Storage/AI-Data/AgentAri/Tokenizer/wordfreq-en-250000/"
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
    using agentari::nn::AdamW;
    using agentari::nn::GenerationConfig;
    using agentari::nn::TransformerConfig;
    using agentari::nn::TransformerModel;

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
        agentari::nn::clip_grad_norm(model.parameters(), 1.0F);
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
    bool test_mode = false;
    bool vulkan_requested = false;
    bool no_gui = false;
    std::size_t total_neuron_count = 0U;
    std::size_t tokenizer_neuron_count = 8000U;
    std::size_t max_cpu_usage_percent = 100U;
    std::size_t max_gpu_usage_percent = 100U;
    std::filesystem::path log_path = "agentari-runtime.log";
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
        } else if (argument == "--test") {
            test_mode = true;
        } else if (argument == "--no-gui") {
            no_gui = true;
        } else if ((argument == "--total-neurons" || argument == "--neurons") &&
                   index + 1 < argc) {
            try {
                const std::string neuron_argument = argv[++index];
                if (!neuron_argument.empty() && neuron_argument.front() == '-') {
                    throw std::invalid_argument("must be positive");
                }
                const auto neurons = std::stoull(neuron_argument);
                if (neurons == 0U ||
                    neurons > static_cast<unsigned long long>(
                                  std::numeric_limits<std::size_t>::max())) {
                    throw std::invalid_argument("must be positive and fit the host size type");
                }
                total_neuron_count = static_cast<std::size_t>(neurons);
            } catch (const std::exception& exception) {
                std::cerr << "Invalid " << argument << " value: " << exception.what() << '\n';
                return 2;
            }
        } else if ((argument == "--tokenizer-neurons" ||
                    argument == "--tokenizer-neuron-count") &&
                   index + 1 < argc) {
            try {
                const std::string tokenizer_argument = argv[++index];
                if (!tokenizer_argument.empty() && tokenizer_argument.front() == '-') {
                    throw std::invalid_argument("must be positive");
                }
                const auto neurons = std::stoull(tokenizer_argument);
                if (neurons == 0U ||
                    neurons > static_cast<unsigned long long>(
                                  std::numeric_limits<std::size_t>::max())) {
                    throw std::invalid_argument("must be positive and fit the host size type");
                }
                tokenizer_neuron_count = static_cast<std::size_t>(neurons);
            } catch (const std::exception& exception) {
                std::cerr << "Invalid " << argument << " value: " << exception.what() << '\n';
                return 2;
            }
        } else if ((argument == "--max-cpu-usage" || argument == "--max-gpu-usage") &&
                   index + 1 < argc) {
            try {
                const std::string usage_argument = argv[++index];
                if (!usage_argument.empty() && usage_argument.front() == '-') {
                    throw std::invalid_argument("must be between 1 and 100");
                }
                const auto usage = std::stoull(usage_argument);
                if (usage == 0U || usage > 100U) {
                    throw std::invalid_argument("must be between 1 and 100");
                }
                if (argument == "--max-cpu-usage") {
                    max_cpu_usage_percent = static_cast<std::size_t>(usage);
                } else {
                    max_gpu_usage_percent = static_cast<std::size_t>(usage);
                }
            } catch (const std::exception& exception) {
                std::cerr << "Invalid " << argument << " value: " << exception.what() << '\n';
                return 2;
            }
        } else if (argument == "--workers" && index + 1 < argc) {
            try {
                const std::string worker_argument = argv[++index];
                if (!worker_argument.empty() && worker_argument.front() == '-') {
                    throw std::invalid_argument("must be zero or positive");
                }
                const auto workers = std::stoull(worker_argument);
                agentari::parallel::set_default_worker_count(
                    static_cast<std::size_t>(workers));
            } catch (const std::exception& exception) {
                std::cerr << "Invalid --workers value: " << exception.what() << '\n';
                return 2;
            }
        } else if (argument == "--log" && index + 1 < argc) {
            log_path = argv[++index];
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

    agentari::parallel::set_max_cpu_usage_percent(max_cpu_usage_percent);
    if (no_gui && !run_for.count()) {
        std::cerr << "--no-gui requires --run-for-ms so the headless process has a bounded run.\n";
        return 2;
    }

    RuntimeTestWorkspace test_workspace(test_mode);
    if (!test_workspace.ready()) {
        std::cerr << "Unable to start test mode: " << test_workspace.error() << '\n';
        return 2;
    }
    if (test_mode) {
        memory_path = test_workspace.file("experiences.tsv");
        std::cout << "Test mode: temporary runtime files in " << test_workspace.root()
                  << " will be removed on deinit.\n";
    }

    agentari::diagnostics::RunLog run_log(log_path);
    const agentari::system::CpuCapabilities startup_cpu =
        agentari::system::cpu_capabilities();
    run_log.info("startup executable=agent-ari test_mode=" +
                 std::string(test_mode ? "true" : "false") +
                 " requested_log=" + log_path.string() +
                 " effective_workers=" +
                 std::to_string(agentari::parallel::default_worker_count()) +
                 " logical_processors=" +
                 std::to_string(startup_cpu.topology.logical_processors) +
                 " usable_processors=" +
                 std::to_string(startup_cpu.topology.usable_processors) +
                 " simd=" + agentari::system::simd_name(startup_cpu.simd) +
                 " max_cpu_usage_percent=" + std::to_string(max_cpu_usage_percent) +
                 " max_gpu_usage_percent=" + std::to_string(max_gpu_usage_percent) +
                 " total_neurons=" + std::to_string(total_neuron_count) +
                 " tokenizer_neurons=" + std::to_string(tokenizer_neuron_count));

    if (neural_demo) {
        run_log.info("running neural demo");
        return run_neural_demo();
    }
    if (word_demo) {
        run_log.info("running word demo");
        return run_word_demo();
    }

    std::optional<agentari::gpu::VulkanComputeBackend> gpu_backend;
    if (vulkan_requested) {
        gpu_backend.emplace();
        if (gpu_backend->available()) {
            agentari::nn::set_tensor_backend(agentari::nn::TensorBackend::vulkan,
                                               &*gpu_backend);
            std::cout << "Tensor backend: Vulkan (" << gpu_backend->device_name() << ")\n";
        } else {
            std::cerr << "Vulkan requested but unavailable: " << gpu_backend->error()
                      << "; using CPU tensors.\n";
        }
    }

    agentari::Memory memory(memory_path);
    std::string error;
    if (!memory.load(error)) {
        run_log.error("failed to load memory: " + error);
        std::cerr << "Failed to load memory: " << error << '\n';
        return 1;
    }

    agentari::ToolRegistry tools;
    if (!tools.register_tool({"time", "Return the local system time.", local_time}, error) ||
        !tools.register_tool({"repeat", "Return the supplied text.", repeat_text}, error)) {
        std::cerr << "Failed to register tool: " << error << '\n';
        return 1;
    }

    agentari::text::LayeredTokenizer tokenizer;
    (void)tokenizer.load_depth_map_file(project_file("data/token-depth-regions.tsv").string(), error);
    if (!tokenizer.load_primitive_file(project_file("data/word-building-rules.tsv").string(),
                                       2048U, error)) {
        for (const std::string primitive : {"aero", "astro", "bio", "chem", "cosmo",
                                             "data", "geo", "mech", "neuro", "system"}) {
            (void)tokenizer.add_primitive(primitive);
        }
    }
    const std::filesystem::path ranked_vocabulary =
        "/var/mnt/Storage/AI-Data/AgentAri/Tokenizer/wordfreq-en-250000/"
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
    agentari::text::PredictorConfig predictor_config;
    predictor_config.maximum_context_words = 256U;
    predictor_config.maximum_sequence_length = 256U;
    predictor_config.model_vocabulary_limit = 8192U;
    agentari::text::WordPredictor predictor(tokenizer, predictor_config);
    agentari::text::ContextSteering context(tokenizer, agentari::text::ContextSteeringConfig{
        .local_window_words = 256U,
        .history_sample_count = 256U,
        .neighborhood_radius = 4U,
        .frame_word_count = 4096U,
        .salient_words_per_frame = 8U,
        .retained_frame_count = 32U,
    });
    agentari::Agent agent(memory, tools, &predictor, &context);
    agentari::AgentWorker agent_worker(agent);
    HierarchicalRuntime hierarchical_runtime =
        start_hierarchical_runtime(total_neuron_count, tokenizer_neuron_count);
    std::optional<agentari::SdlWindow> window;
    if (!no_gui) {
        window.emplace();
        if (!window->open(error)) {
            run_log.error("failed to open SDL window: " + error);
            std::cerr << "Failed to open runtime window: " << error << '\n';
            return 1;
        }
    }

    using Clock = std::chrono::steady_clock;
    constexpr auto ai_period = std::chrono::milliseconds(8);
    constexpr auto render_period = std::chrono::milliseconds(16);
    const auto start = Clock::now();
    auto next_ai_tick = start;
    auto next_render = start;
    std::deque<std::string> pending_commands;
    const agentari::system::CpuCapabilities capabilities =
        agentari::system::cpu_capabilities();
    const agentari::system::GpuCapabilities gpu_capabilities =
        agentari::system::gpu_capabilities();
    std::string status = "Learning worker ready (" +
                         std::string(agentari::system::simd_name(capabilities.simd)) +
                         ", " + std::to_string(capabilities.topology.usable_processors) +
                         " usable CPU threads, " +
                         std::to_string(agentari::parallel::default_worker_count()) +
                         " parallel workers, " +
                         std::to_string(gpu_capabilities.device_count) + " DRM GPU(s)).";
    if (no_gui) {
        status = "Headless learning worker ready (8 ms AI loop; no SDL window).";
    }
    status += " Hierarchical network: " +
              std::to_string(hierarchical_runtime.network.state().total_neuron_count) +
              " neurons (tokenizer=" +
              std::to_string(hierarchical_runtime.network.state().tokenizer_neuron_count) +
              ", general=" +
              std::to_string(hierarchical_runtime.network.state().layer_neuron_count) +
              "; fresh; checkpointing disabled).";
    std::cout << "Hierarchical network: "
              << hierarchical_runtime.network.state().total_neuron_count
              << " neurons (tokenizer="
              << hierarchical_runtime.network.state().tokenizer_neuron_count
              << ", general="
              << hierarchical_runtime.network.state().layer_neuron_count
              << "), spatial cells="
              << hierarchical_runtime.network.spatial_map().occupied_cell_count()
              << " (fresh start; checkpointing disabled)\n";
    run_log.info("runtime ready hierarchical_neurons=" +
                 std::to_string(hierarchical_runtime.network.state().total_neuron_count) +
                 " tokenizer_neurons=" +
                 std::to_string(hierarchical_runtime.network.state().tokenizer_neuron_count) +
                 " layer_neurons=" +
                 std::to_string(hierarchical_runtime.network.state().layer_neuron_count) +
                 " spatial_cells=" +
                 std::to_string(hierarchical_runtime.network.spatial_map().occupied_cell_count()));
    bool input_closed = false;
    bool running = true;
    std::uint64_t ai_ticks = 0;
    std::uint64_t rendered_frames = 0;
    constexpr std::size_t maximum_ai_catch_up_ticks = 4U;

    while (running) {
        bool window_closed = false;
        if (window.has_value()) {
            if (!window->pump_events(pending_commands, window_closed, error)) {
                status = "Input error: " + error;
                window_closed = true;
            }
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
                } else {
                    if (!command.empty() && command.front() != '/') {
                        const auto trace = hierarchical_runtime.network.train_step(
                            hierarchical_runtime.tokenizer, command);
                        status = "Online learner step " +
                                 std::to_string(hierarchical_runtime.network.state().step) +
                                 " updated from " + std::to_string(trace.words.size()) +
                                 " words.";
                        if (!trace.words.empty()) {
                            const auto& last_word = trace.words.back();
                            if (last_word.word !=
                                agentari::text::hierarchical::WordVocabulary::unknown_id) {
                                const auto predictions =
                                    hierarchical_runtime.network.predict_next_word(
                                        last_word.word, 1U);
                                if (!predictions.empty() && !predictions.front().sequence_end) {
                                    const std::string_view next_word =
                                        hierarchical_runtime.tokenizer.words().text(
                                            predictions.front().id);
                                    if (!next_word.empty()) {
                                        status += " Next learned word: " +
                                                  std::string(next_word) + ".";
                                    }
                                }
                            }
                        }
                    }
                    if (!agent_worker.submit(command)) {
                        run_log.debug("agent worker queue full; command deferred");
                        status = "Agent queue is full; input was deferred.";
                    } else {
                        run_log.debug("command submitted to agent worker");
                    }
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
        if (window.has_value() && after_ai >= next_render) {
            ++rendered_frames;
            window->render(agentari::WindowState{
                .uptime = std::chrono::duration_cast<std::chrono::milliseconds>(after_ai - start),
                .ai_ticks = ai_ticks,
                .rendered_frames = rendered_frames,
                .input = window->input_buffer(),
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

        const auto deadline = window.has_value()
                                  ? std::min(next_ai_tick, next_render)
                                  : next_ai_tick;
        const auto current = Clock::now();
        if (deadline > current) {
            const auto wait_duration = deadline - current;
            if (window.has_value()) {
                window->wait_for_next_tick(
                    std::chrono::duration_cast<std::chrono::milliseconds>(wait_duration));
            } else {
                std::this_thread::sleep_for(wait_duration);
            }
        }
    }

    if (window.has_value()) {
        window->close();
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start);
    std::cout << "AgentAri stopped. AI ticks: " << ai_ticks
              << ", rendered frames: " << rendered_frames
              << ", elapsed: " << elapsed.count() << " ms"
              << ", experiences stored: " << memory.size() << '\n';
    run_log.info("shutdown ai_ticks=" + std::to_string(ai_ticks) +
                 " rendered_frames=" + std::to_string(rendered_frames) +
                 " experiences=" + std::to_string(memory.size()));
    return 0;
}
