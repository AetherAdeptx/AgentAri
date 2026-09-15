# AgentAri

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

This early AgentAri research snapshot was created by Ari Stone with Codex AI
(ChatGPT Luna). It has not been tested extensively. Local build, smoke-test, unit-test, and
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

The reproducible development environment is the `agentari-dev` toolbox:

```bash
toolbox run --container agentari-dev \\
  cmake -S /var/home/Ari/Codex/AI/Projects/AgentAri \\
        -B /var/home/Ari/Codex/AI/Projects/AgentAri/build \\
        -DCMAKE_BUILD_TYPE=Debug
toolbox run --container agentari-dev \\
  cmake --build /var/home/Ari/Codex/AI/Projects/AgentAri/build --parallel 2
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
build/agent-ari
```

### One-command setup

The setup script installs the build dependencies, configures the project,
builds it, runs the tests, and launches the SSH-friendly console:

```bash
./tools/install_and_run.sh
```

On systems with Toolbox, it uses the `agentari-dev` development container.
Use `--app` to build and launch the SDL application, or `--no-run` to stop
after a successful build and test pass. Remaining options are passed to the
selected executable:

```bash
./tools/install_and_run.sh --console --workers 0
./tools/install_and_run.sh --console --resume-state \
  --state /absolute/path/to/agentari-state.txt
./tools/install_and_run.sh --app
```

### Feeding text data

The console learns one non-empty input line as one record. Interactive input
can be plain text or an explicit command:

```text
/learn The first sentence becomes a training record.
The next sentence also becomes a training record.
```

For a text file, prefix each line with `/learn` and disable the interactive
picker. Use `--resume-state` when you want to continue an existing state file;
without it, normal startup begins fresh by deleting only the exact state path:

```bash
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] || printf '/learn %s\n' "$line"
done < corpus.txt |
  ./build-agentari/agentari-console --no-prompt \
    --state /absolute/path/to/agentari-training-state.txt
```

To continue that same training state later:

```bash
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] || printf '/learn %s\n' "$line"
done < more-corpus.txt |
  ./build-agentari/agentari-console --no-prompt --resume-state \
    --state /absolute/path/to/agentari-training-state.txt
```

For multiple UTF-8 text files, concatenate their non-empty lines into the
same `/learn` stream before piping it to the console. The state journal is
written after each record, so an interrupted run can be resumed with the same
`--state` path. `/predict TEXT` inspects character, semantic-part, word, and
left-context predictions; `/inspect TEXT` shows the lossless token breakdown.

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

The project now includes a CPU-first neural library in `include/agentari`:

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
- The hierarchical learner amortizes neuron routing across input batches. Its
  demo reports progress and online next-word top-1 accuracy, and supports a
  separate read-only evaluation file. `tools/prepare_oasst_text.py` and
  `tools/prepare_cabnc_text.py` strip dataset metadata while preserving simple
  speaker-role markers before training.
- Batched hierarchical learning uses one persistent delta buffer per active
  worker. Workers collect and coalesce read-only character, part, word, and
  reverse-context evidence; the owning network merges those deltas in worker
  order, so transition tables are never concurrently mutated and repeated
  events use a closed-form bounded update. Single-record online learning stays
  ordered and deterministic.
- `TeacherFeedbackTrainer` provides a frozen-teacher loop: a local small model
  scores AgentAri prompt/candidate pairs, while only AgentAri learns. The
  JSONL bridge is `tools/build_teacher_requests.py` plus
  `tools/qwen_teacher_judge.py` or the GGUF-compatible
  `tools/ollama_teacher_judge.py`; `agentari-teacher-demo` applies the
  bounded scores and high-confidence candidate/correction examples. Both
  judge adapters reject models at or above one billion parameters.
- Vulkan is selectable with `--vulkan`. Tensor rank-2 matmul and matching-shape
  elementwise forward operations use Vulkan when available; reductions,
  attention, and backward accumulation retain checked CPU fallbacks.
- `System.hpp` detects logical/physical/usable CPU counts, cache-line sizing,
  runtime SIMD support, and Linux DRM GPU candidates (including vendor and
  VRAM hints). `Parallel.hpp` dynamically schedules independent work from the
  process CPU affinity, defaults to two fewer usable logical processors than
  detected, supports `--workers N`, and avoids nested oversubscription. The
  worker pool is persistent for the process lifetime, and the read-only
  portion of hierarchical token tracing is parallelized; order-sensitive
  online updates remain deterministic on the owning thread. The same owner
  rule is used when merging thread-local learning deltas.
- `RunLog` is the shared transient diagnostics workflow. `agent-ari` and
  `agentari-teacher-demo` record startup, progress, errors, and shutdown to
  a configured log. On the next startup, only that exact prior log path is
  removed and recreated, so copy it elsewhere first when it needs to be kept.
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
toolbox run --container agentari-dev \
  /var/home/Ari/Codex/AI/Projects/AgentAri/build-ninja/agent-ari --nn-demo
```

Run the tensor, autodiff, attention, optimizer, and generation tests with:

```bash
toolbox run --container agentari-dev \
  ctest --test-dir /var/home/Ari/Codex/AI/Projects/AgentAri/build-ninja \
        --output-on-failure
```

Run the first layered tokenizer and transparent word-prediction baseline with:

```bash
./build-ninja/agent-ari --word-demo
```

The SDL-independent demonstration executable is also available as
`build-ninja/agentari-word-demo` while the SDL development deployment is
pending.

The word demo loads the first 100,000 entries from the ranked vocabulary when the configured
mechanical-drive dataset is present, otherwise it uses a tiny built-in fallback.
Its context layer exposes the 256-word local window, weighted historical
sampling, frame summaries, and goal input. The predictor uses the learned
Transformer context layer with an 8,192-word CPU shortlist by default. Grammar
layer 5 now repairs spacing, sentence capitalization, punctuation, and explicit
exceptions; a full probabilistic grammar parser remains future work.

Run the in-memory hierarchical learner with:

```bash
./build-agent/agentari-hierarchical-demo
./build-agent/agentari-hierarchical-demo --train /path/to/text.txt --passes 1 --prompt "the quick"
```

For a clean OpenAssistant slice and a held-out evaluation file:

```bash
python3 tools/prepare_oasst_text.py /path/to/oasst1-all.messages.jsonl.gz \
  --limit 800 --output /tmp/agentari-train.txt
python3 tools/prepare_oasst_text.py /path/to/oasst1-all.messages.jsonl.gz \
  --skip 800 --limit 200 --output /tmp/agentari-eval.txt
./build-agent/agentari-hierarchical-demo \
  --train /tmp/agentari-train.txt --eval /tmp/agentari-eval.txt \
  --passes 1 --batch-size 64 --progress-every 100 \
  --prompt "user Can you write a short introduction"
```

This demo trains as it processes each text chunk, then immediately queries
sequence-start, character, part, word, and reverse-context predictions. It does
not read or write checkpoints; all learned state disappears when the process
ends.

For an interactive SSH session, build and run the SDL-independent console:

```bash
cmake -S . -B build-console -DAGENTARI_BUILD_APP=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build build-console --target agentari-console --parallel 2
./build-console/agentari-console --workers 0
```

The console starts a fresh 56,000-neuron, 28-layer network in memory by
default. Use
`/learn TEXT`, `/predict TEXT`, `/inspect TEXT`, `/feedback QUALITY CONF TEXT`,
`/state`, and `/context`; plain text is shorthand for `/learn TEXT`. It is a
terminal executable, not a network daemon, so the SSH workflow is simply to
log into the host and run the binary. Checkpointing is disabled and the
transient `agentari-console.log` is recreated on each start. Learned records
are appended to `agentari-console-state.txt` as they arrive; normal startup
deletes that exact state file, while `--resume-state` bypasses deletion and
replays it to reconstruct the tokenizer and network weights. Use `--state PATH`
to select another state file.

When `/predict` is run from a real terminal, it requests up to 64 learned word
candidates and opens a picker. Use Up/Down or `w`/`s` to move, Tab to advance,
Enter to select, and Esc or `q` to cancel. A pipe or `--no-prompt` prints the
same candidate list without entering raw-terminal mode.

For frozen-model judging rather than fine-tuning the small model:

```bash
python3 tools/build_teacher_requests.py /path/to/oasst1-all.messages.jsonl.gz \
  --limit 8 --output /tmp/agentari-teacher-requests.jsonl
python3 tools/qwen_teacher_judge.py --model /path/to/transformers-model \
  --threads 4 < /tmp/agentari-teacher-requests.jsonl \
  > /tmp/agentari-teacher-feedback.jsonl
./build-training-test/agentari-teacher-demo \
  --feedback /tmp/agentari-teacher-feedback.jsonl --max-records 8 --neurons 4000 \
  --workers 0 --log /tmp/agentari-teacher-run.log
```

The judge is loaded locally with gradients disabled and never receives an
optimizer. Its output is clamped and confidence-gated by AgentAri; the loop
updates only the in-memory AgentAri tokenizer/predictor. Checkpoints remain
disabled, so all learned state is discarded when the process exits.

The teacher executable uses the existing affinity-aware worker policy. `--workers 0`
means usable logical CPUs minus two, while a positive value caps the worker
count explicitly. Tokenization analysis uses those workers where records are
large enough; mutation and feedback application stay ordered because this is
an online learner. Both executables use transient logs by default
(`agentari-training.log` or `agentari-runtime.log` in the current run
directory), with progress lines emitted during long runs.
Each teacher record logs its valid record number and source line number.
The current teacher defaults require an overall candidate score of at least
0.86 and confidence of at least 0.90 before an answer is learned positively;
corrections use the same stricter defaults. The semantic-part learner indexes
parts by their first byte, reuses ordered scratch buffers, and refreshes only
parts touched by feedback.

For the MSI's local Qwen3.5 0.8B GGUF through Ollama:

```bash
python3 tools/ollama_teacher_judge.py \
  --host http://192.168.254.214:11434 --model qwen3.5:0.8b \
  < /tmp/agentari-teacher-requests.jsonl \
  > /tmp/agentari-teacher-feedback.jsonl
```

Build and play with the interactive toy:

```bash
cmake -S . -B build-gpu -DAGENTARI_BUILD_APP=OFF
cmake --build build-gpu --target agentari-toy
./build-gpu/agentari-toy
```

The toy is also packaged on the desktop as `AgentAri Toy.desktop`. It
supports `/learn`, `/predict`, `/generate`, `/goal`, `/history`, `/state`,
`/feedback`, `/save`, `/load`, and `/gpu`.

The detailed architecture notes are in `docs/NEURAL_ARCHITECTURE.md`.

## Run

```bash
./build/agent-ari
```

For a deterministic runtime smoke test without opening a visible display:

```bash
SDL_VIDEODRIVER=dummy ./build/agent-ari --run-for-ms 200
```

To opt the tensor forward path into Vulkan when the configured backend is
available:

```bash
SDL_VIDEODRIVER=dummy ./build/agent-ari --vulkan --run-for-ms 200
```

For an isolated runtime smoke test, use `--test`. It creates a uniquely named
temporary workspace for runtime files and removes that workspace on deinit;
the hierarchical 56,000-neuron network (28 layers x 2,000 neurons) also starts fresh because checkpointing
is currently disabled while the primitive layers are being designed:

```bash
SDL_VIDEODRIVER=dummy ./build-agent/agent-ari --test --run-for-ms 200
```

The default experience file is `data/experiences.tsv`. A different path can be
selected without changing source code:

```bash
./build/agent-ari --memory /absolute/path/to/experiences.tsv
```

## First commands

```text
/help
/remember The agent should explain its reasoning clearly.
/recall
/recall reasoning
/use time
/use repeat hello from the AgentAri
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
