# AgentAri

AgentAri is Ari Stone’s from-scratch C++ experiment in self-learning,
hierarchical tokenization, and agentic computation. It is an early research
prototype created with Codex AI—not a pretrained LLM or production system.

## What exists

- C++20 neural and tensor foundations with automatic differentiation.
- Hierarchical tokenizer layers for characters, semantic parts, words, and
  context.
- Online learning for character, part, word, and reverse-context transitions.
- A configurable neuron network with recurrent state, feedback, SIMD, and
  multithreaded CPU execution.
- Decoder-style Transformer components, AdamW, RMSNorm, RoPE, attention, and
  KV-cache experimentation.
- Optional Vulkan compute backend; CUDA is intentionally not required.
- SDL3 application shell and an SSH-friendly interactive console.
- MIT licensed.

The current word predictor is intentionally simple. It learns from observed
text and transition evidence, but it does not yet have the contextual depth or
general language ability of modern open-source language models.

## Quick start

From the project directory:

```bash
./tools/install_and_run.sh
```

The script installs dependencies in the `agentari-dev` Toolbox when available,
configures and builds the project, runs all tests, and launches the console.

Useful options:

```bash
./tools/install_and_run.sh --no-run       # install, build, and test only
./tools/install_and_run.sh --app          # build and launch the SDL app
./tools/install_and_run.sh --console --workers 0
```

Manual CMake builds are also supported:

```bash
cmake -S . -B build-agentari -G Ninja \
  -DAGENTARI_BUILD_APP=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-agentari --parallel
ctest --test-dir build-agentari --output-on-failure
```

Code::Blocks users can open [`AgentAri.cbp`](AgentAri.cbp). Its targets
configure and build the project through CMake, including the console, tests,
and SDL app.

## Console use

Start the console with:

```bash
./build-agentari/agentari-console --workers 0
```

Then enter:

```text
/learn The system learns one text record.
/predict The system
/inspect semantic parts and atomic tokens
/state
/context
/quit
```

Plain text is treated as `/learn`. `/predict` displays up to 64 candidate
words; use Up/Down and Enter to select one in a real terminal.

Learned records can persist between runs in a state journal:

```bash
./build-agentari/agentari-console \
  --resume-state --state /absolute/path/agentari-state.txt
```

Without `--resume-state`, the exact state path is cleared on startup so an
experiment begins fresh.

The network can load a mix of the available compile-time neuron presets at
runtime. Each entry stores a neuron type and percentage fill:

```cpp
config.neuron_mix.add(agentari::neuron::NeuronType::basic, 5.0F);
config.neuron_mix.add(agentari::neuron::NeuronType::modern_gated, 4.0F);
```

Partial mixes leave the remainder explicitly unassigned. A complete mix uses
largest-remainder rounding so its type counts add up exactly to the network
budget; `/state` prints the resulting counts.

## Feeding a text file

Each non-empty line becomes one training record. This example preserves lines
that begin with punctuation or a slash:

```bash
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] || printf '/learn %s\n' "$line"
done < corpus.txt |
  ./build-agentari/agentari-console --no-prompt \
    --state /absolute/path/agentari-training-state.txt
```

Add `--resume-state` to continue an existing journal. Keep downloaded corpora,
model weights, logs, and build directories outside Git.

## Project layout

`include/agentari/` contains public headers. `src/` contains implementation.
`tools/` contains the console, demos, data preparation, teacher-feedback, and
setup scripts. `tests/` contains the current regression suite. `data/` contains
small tokenizer maps and rules.

## Provenance

Created by Ari Stone with Codex AI (ChatGPT Luna). This project has limited
testing and changing interfaces; expect incomplete features and bugs.

License: [MIT](LICENSE)
