# Neural architecture

This document describes the first serious neural core of FirstAgent. It is
intentionally small enough to inspect, but its boundaries are chosen so a
larger model and accelerated backends can be added without rewriting the
agent runtime.

## Data flow

```text
token ids
   │
   ▼
token embedding ───────────────────────────────┐
   │                                            │
   ▼                                            │ tied output projection
Transformer block × N                           │
   ├─ RMSNorm                                    │
   ├─ Q/K/V projections                          │
   ├─ RoPE                                       │
   ├─ causal grouped-query attention             │
   ├─ residual connection                         │
   ├─ RMSNorm                                    │
   ├─ SwiGLU feed-forward                         │
   └─ residual connection                         │
   │                                            │
   ▼                                            │
final RMSNorm ──────────────────────────────────┘
   │
   ▼
logits over vocabulary
```

## Tensor layer

`Tensor` owns a contiguous `float` buffer and a shape. Copies share the same
implementation, which means a parameter can be passed through many operations
and still receive one accumulated gradient buffer.

The first implementation uses row-major layout. It supports scalar
broadcasting and final-dimension broadcasting, which is enough for biases and
normalization weights in the current model. Operations reject incompatible
shapes rather than silently resizing data.

Independent tensor rows, elementwise ranges, attention query ranges, layout
transforms, optimizer parameter ranges, and safe gradient ranges use the shared
affinity-aware scheduler. Its automatic worker count is the usable logical CPU
count minus two, clamped to at least one; callers can override it globally or
with the application `--workers N` option. Reductions use private partials so
parallel accumulation does not create data races. Backward paths that would
collide on shared parameters, such as repeated-token embedding updates, remain
serialized until they gain a reduction buffer.

## Configurable neuron layer

`Neuron.hpp` is a separate header-only component for combining different
neuron families in a future heterogeneous network. `AdaptiveNeuron` takes
compile-time input, recurrent-state, and output widths plus activation,
storage, and feature policies. Feature switches can compile out recurrent
weights, message-feedback weights, gates, residual mixing, normalization, and
eligibility traces. Storage policies provide:

- aligned FP32 arrays for learning and maximum CPU throughput;
- int8 weights for compact inference-oriented banks;
- one-bit sign weights for extremely compact binary inference paths.

The built-in presets range from `BasicNeuron` to `ModernGatedNeuron`, with
SiLU/GELU, gated residual state, message feedback, normalization, and traces.
`SparseModernNeuron` adds compile-time top-k mixture-of-experts routing. The
class reports compile-time dimensions and object size so a network can mix
small fast neurons with larger recurrent, sparse, or compact specialist neurons
without making the whole network use one storage policy.

Autograd records a small operation node containing parent storage and a
backward function. Calling `loss.backward()` walks the graph in reverse
topological order and accumulates gradients into leaf tensors. `NoGradGuard`
disables graph construction during generation and other inference work.

## Transformer choices

The model is a decoder-only language model suitable for causal next-token
prediction:

- RMSNorm keeps normalization simple and avoids a learned bias.
- RoPE encodes relative token position by rotating feature pairs.
- Grouped-query attention uses more query heads than key/value heads, reducing
  key/value cache size while retaining multiple query projections.
- Causal scaled dot-product attention prevents a position from reading future
  positions.
- SwiGLU gives the feed-forward block a learned value path and a learned gate.
- Residual connections preserve the existing representation while each block
  learns an update.
- The output projection is tied to the input embedding matrix, reducing
  parameters and coupling input/output token representations.
- Full-sequence projections are explicitly rearranged from row-major token
  layout into head-major attention layout; tests compare those logits with the
  incremental path.
- Incremental generation stores rotated keys and values per layer in a KV
  cache, so each new token attends to the existing history without recomputing
  all previous projections.

## Training boundary

`TransformerModel::loss` accepts an input sequence and an equally sized target
sequence. It returns mean cross-entropy over the vocabulary. `AdamW` owns first
and second moments for the model parameters, and gradient clipping is exposed
as a separate operation so an agent policy can choose its training rules later.

The current model is intentionally a research kernel, not a production LLM:
it has no dataset batching, dropout,
mixed precision, distributed training, or full Transformer GPU execution yet. Those should be
added behind the existing boundaries and measured against the CPU reference.

## Planned extension points

1. Add a `DType` and storage/backend abstraction while retaining FP32 as the
   reference implementation.
2. Add batched tensor operations and a data-loader interface.
3. Add checkpoint metadata for tokenizer vocabulary/configuration hashes.
4. Route the existing Vulkan matrix backend into matrix multiplication, then add normalization and
   attention, comparing every kernel against the CPU implementation.
5. Add tokenizer and dataset interfaces without coupling them to SDL.
6. Add dropout, learning-rate schedules, validation metrics, and checkpoint
   recovery after the reference training loop is stable.
