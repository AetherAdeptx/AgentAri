# FirstAgent project plan

## Design principles

- Keep the first executable understandable enough to inspect line by line.
- Separate memory, decisions, and tools so each can evolve independently.
- Treat external actions as dangerous by default and use an explicit allow-list.
- Record outcomes so future learning can be evaluated instead of guessed.
- Keep model integration optional until the core agent protocol is stable.
- Keep tensor operations behind a backend-neutral interface.
- Prefer open, portable compute standards; do not make CUDA a project
  dependency.

## First chunk scope

The first chunk establishes a runnable Linux executable with:

- a command-line interaction loop;
- a monotonic real-time scheduler with an 8 ms AI tick;
- a replaceable terminal window/rendering layer;
- append-only, escaped TSV experience storage;
- recall by recency or text search;
- a small registry of deterministic tools;
- a clean boundary between the agent and tool implementations.

## Not in scope yet

- autonomous unrestricted shell commands;
- network access;
- unbounded or unmeasured model training;
- pretending stored text is understanding;
- hidden background activity.

## Planned compute foundation

The first learning implementation should run on the CPU so its behavior can
be tested independently of hardware. When acceleration is justified, the
preferred backend is Vulkan compute with GLSL/SPIR-V. OpenCL may be added as a
second backend for portability. CUDA is out of scope.

## Neural core status

The first CPU reference implementation is now present. It includes:

- contiguous tensors with shared storage and reverse-mode autodiff;
- matrix operations, stable softmax and cross-entropy;
- embeddings, RMSNorm, RoPE, and causal attention;
- a decoder-only Transformer with grouped-query attention and SwiGLU;
- tied embeddings, AdamW, gradient clipping, and KV-cache generation;
- executable tests for tensor math, gradients, model shapes, optimization, and
  decoding.

The predictor now has a versioned binary checkpoint containing model weights,
AdamW moments, and tokenizer learning-node state. The next work should be
measurement-driven: verify learning curves and resume behavior on a tiny
dataset, then introduce batching and an accelerated backend one operation at a
time.

## Questions for our next design pass

- What should count as a successful outcome?
- Should memories have confidence, importance, or expiration?
- What is the smallest useful goal/action/feedback cycle?
- Do we want a symbolic learner first, a neural learner first, or both?
- Which actions require Ari's approval every time?
