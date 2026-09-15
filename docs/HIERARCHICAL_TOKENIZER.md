# Experimental four-layer tokenizer

The experimental tokenizer is intentionally separate from the original
five-layer `LayeredTokenizer` so the existing predictor remains compatible
while the real-part experiment evolves.

```text
characters, punctuation, numbers
              ↓
       learned contiguous parts
              ↓
          whole words
              ↓
           context state
```

The atomic layer has no vocabulary and no precomputed word pieces. It emits
one byte-sized atomic token per input byte and classifies it as a character,
punctuation, number, whitespace, or other byte. Word boundaries are spans over
those atomic tokens, so the original input can always be reconstructed.

The semantic-part learner starts with an empty part bank. It counts repeated
contiguous spans across distinct observed words and promotes reusable spans
into stable IDs. A part analysis uses dynamic programming to choose learned
spans while retaining atomic character fallbacks. The current promotion score
is deliberately simple and inspectable; it is not being presented as a
finished morphological or semantic model.

The word vocabulary is built only from observed complete words. It assigns
stable IDs above the atomic and part layers and tracks observations plus an
optional external frequency rank. The context tokenizer consumes those word
IDs, retains a bounded recent window, and currently learns a compact bigram
transition table as the first context baseline.

The learner now records neighboring-word evidence for every candidate. A part
gets a contextual-support score based on how consistently its carrier words
share nearby words, and a compositionality score combining reuse across word
types with that contextual support. These are inspectable learning signals,
not a claim that the current heuristic has solved morphology or semantics.

The next research steps are to add background-frequency correction, compare a
part's context against the contexts of its complete carrier words, replace the
heuristic promotion score with a trainable compositionality loss, and add
held-out tests for meaningful decomposition versus accidental substrings and
exceptions.

## Frozen-teacher feedback loop

`TeacherFeedbackTrainer` is the first bridge to a small external model. It
does not fine-tune, backpropagate through, or save the judge model. A local
Qwen/SmolLM-style model receives a quoted prompt and candidate and emits
bounded JSONL scores for segmentation, semantics, grammar, context relevance,
and confidence. AgentAri clamps those values before applying them.

The loop is:

```text
prompt → AgentAri observes and learns
       → candidate is generated elsewhere
       → frozen local judge scores candidate/correction
       → bounded feedback updates existing associations
       → high-confidence good candidates/corrections become AgentAri examples
```

Positive candidate learning is gated by both an overall quality threshold and
judge confidence. A correction has its own gates. Negative feedback can adjust
the bounded teacher bias of associations that already exist, but feedback
cannot allocate arbitrary transitions or parts. This is intentional: the
teacher is an evaluator and source of training signals, not the owner of
AgentAri's state.

The data and runtime adapters are:

```bash
python3 tools/build_teacher_requests.py /path/to/oasst.jsonl.gz \
  --limit 8 --output /tmp/agentari-teacher-requests.jsonl
python3 tools/qwen_teacher_judge.py \
  --model /path/to/transformers-model \
  --threads 4 < /tmp/agentari-teacher-requests.jsonl \
  > /tmp/agentari-teacher-feedback.jsonl
./build-training-test/agentari-teacher-demo \
  --feedback /tmp/agentari-teacher-feedback.jsonl --max-records 8 --neurons 4000
```

The Qwen path must point to a Transformers-format local directory, not a GGUF
file. The Python runner requires a separately prepared PyTorch/Transformers
environment. It uses `local_files_only=True`, inference mode, and deterministic
decoding; missing dependencies or malformed model output produce neutral
feedback instead of silently training on a guess.

For the MSI's local GGUF model, use the standard-library Ollama adapter:

```bash
python3 tools/ollama_teacher_judge.py \
  --host http://192.168.254.214:11434 --model qwen3.5:0.8b \
  < /tmp/agentari-teacher-requests.jsonl \
  > /tmp/agentari-teacher-feedback.jsonl
```

Both judge adapters enforce a strict parameter limit below one billion. The
MSI validation used Qwen3.5 0.8B, reported as 873,438,784 parameters. The
model only evaluates examples; AgentAri is the component that learns.

The C++ JSON reader is deliberately limited to this controlled output schema,
not a general-purpose JSON implementation. Replace it with the project's
structured JSON dependency before accepting untrusted or arbitrary JSON.

## Predictive layer scaffold

`HierarchicalPredictionNetwork` adds four sparse associative predictor banks.
Each call to `train_step()` performs tokenization, activation routing, and
association learning together, so normal use is also an online training pass:

```text
atomic character → next atomic character
learned part      → next part
whole word        → next word
current word      → word immediately to the left
```

Each learned association is an adaptive prediction neuron with a bounded
weight, activation, observation count, and stable transition key. The current
runtime intentionally keeps this state in memory only; checkpoint serialization
is reserved for a later experiment after the primitive layers stabilize.

Each layer reserves 15% of its neurons for local input and 85% for the outer
network-reference bank. Ten percent of that outer bank currently receives
active cross-layer feedback; the remaining outer capacity is reserved for
future network-wide references. A single total-neuron budget controls the
whole network, and relative layer-size weights automatically allocate that
budget across the four tokenizer layers and any additional layers while
preserving the exact total. The default network has 24 additional layers,
making 28 layers total at 2,000 neurons each.

Every layer also receives `min_neurons_per_layer` before the remaining global
budget is distributed. The default project-wide floor is exposed as
`Min_Neurons_Per_Layer`. The remainder follows the configured layer weights;
ties are assigned in a symmetric outside-in order so rounding does not always
favor the first layer.

## Spatial neuron map

`SpatialNeuronMap` gives the allocated layer banks a physical-style coordinate
system without pretending that ordinary memory addresses are biological
locations. It uses a fixed `128 x 128 x 128` grid with a dense occupancy index
and per-cell linked buckets, so a cell can hold multiple neurons if a later
compressed or over-subscribed representation needs it.

Layer anchors use a three-dimensional Fibonacci-sphere placement with a
golden-angle rotation. The layer spacing term grows with Fibonacci numbers.
Neuron tiles are placed around each anchor; coordinates reflect at the grid
boundary, and collisions first probe by a left-rotating golden-angle spiral
before using a deterministic full-grid fallback.

`smart_search()` returns nearest-first materialized results. `lazy_search()`
walks the occupied grid cells through a callback, supports a layer filter and
result limit, and can stop without allocating a result list. This is a
geometry and locality experiment only; the current allocator does not claim
that neighboring coordinates automatically create useful semantic structure.
