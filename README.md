# FirstAgent

The first C++ chunk of our from-scratch Linux AI project.

This is intentionally a small foundation rather than a pretend finished AI. It
provides small pieces we can develop independently:

- an interactive agent loop;
- durable experience memory;
- an allow-listed tool registry for controlled agentic actions.
- an asynchronous agent worker so model/tool work cannot block SDL rendering;
- a header-only CPU capability, alignment, bit-packing, and affinity-aware
  parallel execution layer.

The current self-learning behavior has two bounded forms: the agent records
experiences for recall, and the word predictor performs small online AdamW
updates on observed text. It is still a research prototype, not a pretrained
general-purpose LLM or a claim of general intelligence.

## Project status and provenance

This early research snapshot was created collaboratively with ChatGPT Luna.
It has not been tested extensively. Local build, smoke-test, unit-test, and
sanitizer checks exist, but the project is not production-ready and should be
expected to contain bugs, incomplete features, and changing interfaces.

Local build directories, downloaded dependency caches, and machine-specific
data are intentionally excluded from the repository.

## Build

From this directory, after SDL3 development files are available:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

The reproducible development environment is the `firstagent-dev` toolbox:

```bash
toolbox run --container firstagent-dev \\
  cmake -S /var/home/Ari/Codex/AI/Projects/FirstAgent \\
        -B /var/home/Ari/Codex/AI/Projects/FirstAgent/build \\
        -DCMAKE_BUILD_TYPE=Debug
toolbox run --container firstagent-dev \\
  cmake --build /var/home/Ari/Codex/AI/Projects/FirstAgent/build --parallel 2
```

The toolbox also contains the AI-development foundation: Eigen 5 for
linear algebra, xtensor 0.27 for multidimensional arrays, OpenBLAS for an
optional optimized CPU backend, SQLite and JsonCpp for structured state, and
Catch2 for tests. GDB and Ninja are available for debugging and builds. SDL3,
SDL3_image, and SDL3_ttf are available for the application layer.

CUDA is intentionally not part of this project. The GPU path is
Vulkan compute, using GLSL compiled to SPIR-V, with OpenCL as an optional
fallback. This keeps the tensor interface portable and avoids coupling the AI
to a proprietary CUDA API. The active NVIDIA hardware driver on this system is
still vendor-provided; the application-side compute interface remains
vendor-neutral.

If CMake is not available, the included standard Makefile provides the same
basic build once SDL3 development files are installed:

```bash
make
```

On the current Bazzite system, `cmake` and `SDL3-devel` are staged in the next
deployment. They are not active until the system is rebooted.

The executable is:

```text
build/first-agent
```

The runtime uses a monotonic fixed-step scheduler. AI work is dispatched every
8 ms (125 Hz), while the SDL renderer redraws every 16 ms (about 60 Hz).
SDL events are pumped in the main loop, so window close, keyboard input, AI
work submission, response polling, and rendering share one controlled runtime.
The predictor and tool calls run on the agent worker rather than on the SDL
thread.

SDL is the active cross-platform windowing backend. Raylib is intentionally not
used by this first runtime; it remains a future option for rapid experiments.
SDL3_image and SDL3_ttf are installed in the development toolbox for a later
asset/text-rendering pass, but are not linked yet.

## Neural core

The project now includes a CPU-first neural library in `include/firstagent`:

- `Tensor` uses contiguous row-major storage and shared handles.
- `kernel::Matrix<Scalar, Backend, Layout>` provides a policy-based matrix
  layer for reusable scalar/layout/backend specializations. The existing
  autograd `Tensor` uses the CPU policy through a compatibility bridge.
- Reverse-mode autodiff supports tensor arithmetic, matrix multiplication,
  activations, softmax, cross-entropy, embeddings, RMSNorm, RoPE, and causal
  scaled dot-product attention.
- The Transformer is decoder-only and uses tied token embeddings, RMSNorm,
  rotary position embeddings, grouped-query attention, SwiGLU feed-forward
  blocks, residual paths, AdamW, gradient clipping, and incremental KV-cache
  generation.
- Predictor training now uses bounded overlapping windows, gradient accumulation,
  streamed file batches, and evaluation loss. The model remains intentionally
  small and online-trainable rather than a pretrained general-purpose LLM.
- Vulkan is selectable with `--vulkan`. Tensor rank-2 matmul and matching-shape
  elementwise forward operations use Vulkan when available; reductions,
  attention, and backward accumulation retain checked CPU fallbacks.
- `System.hpp` detects logical/physical/usable CPU counts, cache-line sizing,
  runtime SIMD support, and Linux DRM GPU candidates (including vendor and
  VRAM hints). `Parallel.hpp` dynamically schedules independent work from the
  process CPU affinity, defaults to two fewer usable logical processors than
  detected, supports `--workers N`, and avoids nested oversubscription.
- `BitPacking.hpp` provides checked bit-width math, packed vectors, and
  LSB-first readers/writers. `AlignedPackedArray<Bits, Alignment>` adds
  cache-line-blocked packed storage for hot categorical/state arrays. Tokenizer
  region states use two bits each, and predictor checkpoints are versioned,
  fingerprinted, and installed through a temporary-file rename.
- `Neuron.hpp` provides compile-time-sized adaptive neurons. Feature policies
  select recurrent state, message feedback, gates, residuals, normalization,
  and eligibility traces; storage policies select FP32, int8, or one-bit
  weights. `BasicNeuron`, `ModernGatedNeuron`, `CompactModernNeuron`, and
  `BinaryInferenceNeuron` are starting presets; `SparseModernNeuron` adds
  compile-time top-k mixture-of-experts routing. The underlying template
  dimensions remain open for mixed neuron networks.
- The layered tokenizer has bounded adaptive learning-node banks for ASCII,
  word primitives, words, context, and grammar. Each bank exposes compact
  activation summaries, routed experts, learning factors, and recurrent and
  message feedback paths. Unknown words and symbols use the deeper lossless
  layers; known words normally stay in the shallow word/context path.

Run the neural smoke demonstration with:

```bash
toolbox run --container firstagent-dev \
  /var/home/Ari/Codex/AI/Projects/FirstAgent/build-ninja/first-agent --nn-demo
```

Run the tensor, autodiff, attention, optimizer, and generation tests with:

```bash
toolbox run --container firstagent-dev \
  ctest --test-dir /var/home/Ari/Codex/AI/Projects/FirstAgent/build-ninja \
        --output-on-failure
```

Run the first layered tokenizer and transparent word-prediction baseline with:

```bash
./build-ninja/first-agent --word-demo
```

The SDL-independent demonstration executable is also available as
`build-ninja/firstagent-word-demo` while the SDL development deployment is
pending.

The word demo loads the first 100,000 entries from the ranked vocabulary when the configured
mechanical-drive dataset is present, otherwise it uses a tiny built-in fallback.
Its context layer exposes the 256-word local window, weighted historical
sampling, frame summaries, and goal input. The predictor uses the learned
Transformer context layer with an 8,192-word CPU shortlist by default. Grammar
layer 5 now repairs spacing, sentence capitalization, punctuation, and explicit
exceptions; a full probabilistic grammar parser remains future work.

Build and play with the interactive toy:

```bash
cmake -S . -B build-gpu -DFIRSTAGENT_BUILD_APP=OFF
cmake --build build-gpu --target firstagent-toy
./build-gpu/firstagent-toy
```

The toy is also packaged on the desktop as `FirstAgent Toy.desktop`. It
supports `/learn`, `/predict`, `/generate`, `/goal`, `/history`, `/state`,
`/feedback`, `/save`, `/load`, and `/gpu`.

The detailed architecture notes are in `docs/NEURAL_ARCHITECTURE.md`.

## Run

```bash
./build/first-agent
```

For a deterministic runtime smoke test without opening a visible display:

```bash
SDL_VIDEODRIVER=dummy ./build/first-agent --run-for-ms 200
```

To opt the tensor forward path into Vulkan when the configured backend is
available:

```bash
SDL_VIDEODRIVER=dummy ./build/first-agent --vulkan --run-for-ms 200
```

The default experience file is `data/experiences.tsv`. A different path can be
selected without changing source code:

```bash
./build/first-agent --memory /absolute/path/to/experiences.tsv
```

## First commands

```text
/help
/remember The agent should explain its reasoning clearly.
/recall
/recall reasoning
/use time
/use repeat hello from the first agent
/stats
/quit
```

Only tools registered in `src/main.cpp` can be executed. There is deliberately
no arbitrary shell execution in this first chunk.

## Planned next chunks

1. Add structured observations, goals, actions, and outcomes.
2. Add a small planner that selects from known actions.
3. Add dataset batching, validation metrics, and learning-rate schedules.
4. Route validated Tensor operations through Vulkan incrementally.
5. Add permissions, approval rules, and recovery for real-world tools.

Vulkan is the current open compute integration target and remains optional at
configure time. The CPU path and scalar fallback remain available on systems
without the Vulkan SDK or a compatible GPU.
