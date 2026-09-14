# Layered tokenizer and first word predictor

The first predictor follows the five layers proposed for FirstAgent:

| Layer | Role | Initial implementation |
| --- | --- | --- |
| 1 | ASCII base | 256 active byte tokens plus 256 reserved growth slots |
| 2 | Word-building rules | 1,024 active rule slots plus 1,024 reserved expansion slots |
| 3 | Complete words | 100,000 frequency-ranked words, with explicit capacity tracking |
| 4 | Context steering | 256 recent words, 256 weighted history samples, frame summaries, and goal input |
| 5 | Grammar | Bounded structural stream of words and punctuation; rule learning remains open |

Layer-1 IDs 0–255 are the active ASCII bytes; IDs 256–511 are reserved for
future letter/byte expansion. Layer-2 IDs are reserved separately from word IDs
so adding word-building rules does not renumber the complete-word vocabulary.
The current `data/word-building-rules.tsv` seed file contains initial semantic
construction rules, while the
implementation reserves 1,024 rule slots and 1,024 expansion slots from the
start. Layer 3 loads the first 100,000 entries from the larger ranked word list
and leaves capacity accounting visible for later additions.

## Current behavior

`LayeredTokenizer::encode_ascii` provides the lossless byte fallback. The
word-level encoder extracts ASCII words and apostrophes, normalizes them to
lowercase, and maps unknown words to `<unk>` for prediction. The separate
`encode_word_parts` path keeps unknown spelling lossless by using the longest
registered primitive at each position and layer-1 ASCII bytes for residual
characters. This is a deterministic starting rule, not a claim that the
current primitive list infers true etymological roots.

`ContextSteering` stores chat entries by log number. It builds a local window of
the most recent 256 words, samples 256 historical slots with a density bias
toward the recent past, searches each selected slot within four words in each
direction, and keeps the least-common ranked word found there. Each 4,096-word
frame contributes up to eight salient words, with the most recent 32 frames
retained. A goal can be stored and loaded by the same log number.

`WordPredictor` is now the learned context layer. It uses the project's
decoder-only Transformer, RMSNorm, RoPE, grouped-query attention, SwiGLU,
weight-tied output projection, and AdamW optimizer. The sparse tokenizer IDs
remain stable for the layered map, while the complete-word tokens are remapped
to a dense, frequency-ranked model vocabulary; this avoids allocating logits for
ASCII and reserved rule slots. The first CPU runtime activates a configurable
frequent-word shortlist (8,192 by default) while retaining all 100,000 tokenizer
entries; this is an explicit performance boundary until sampled/adaptive
softmax is implemented. `observe_file` streams bounded token batches into
overlapping training windows with optional gradient accumulation and byte
limits, so large collections can be sampled without loading the entire source
into memory. `evaluate_loss` provides a no-gradient validation measurement.
Prediction consumes the flattened
goal/frame/history/recent context and returns only complete-word classes.
The default learned context limit is 1,024 words, enough for the recent window,
sampled history, retained frame summaries, and a goal; it can be reduced for
small realtime experiments.
The active model vocabulary is reported separately from the tokenizer capacity.

## Adaptive tokenizer neuron stack

`TokenizerNeuralStack` adds bounded learning-node banks for all five layers.
Known words normally activate the shallow word/context banks. Symbols and
unknown words additionally activate the ASCII and primitive banks through the
lossless word-part path. `encode_text`/`decode_text` add a mixed stream that
preserves whitespace and symbols while still using shallow word tokens. The
grammar bank receives word boundaries plus punctuation/symbol tokens. The
`GrammarEngine` learns transition statistics, accepts explicit exception rules,
normalizes spacing, capitalizes sentence starts, and can add terminal
punctuation to generated text. It remains a compact deterministic repair layer
rather than a full probabilistic parser.

Each bank has a configurable learning factor, fast and slow prototypes,
confidence, activation traces, and four routed expert groups. Every update
produces a compact primitive state containing total activation, center of mass,
entropy, novelty, uncertainty, upward/downward signals, prediction error,
message strength, and sixteen low-resolution region states. These summaries
are the hardwired state bus shared by the tokenizer layers.

The stack has two feedback paths. The recurrent path carries higher-layer
features back toward lower layers on the next cycle. The message path accepts
external feedback from an agent response and modulates the next tokenizer
update. Prediction loss is also fed back as a normalized error signal. This is
an adaptive functional analogue of biological integration, without simulating
spikes, membrane voltages, or other biological mechanisms.

Tensor matrix multiplication uses a runtime-dispatched AVX2/FMA dot-product
path when the CPU supports it, with a scalar fallback for portability. Larger
CPU matrix operations, elementwise transforms, normalization, attention
forward work, layout transforms, optimizer updates, and safe gradient paths
schedule independent ranges across the configured worker set; reductions use
per-worker partials where necessary. Small operations stay serial to avoid
thread overhead, and nested kernels fall back locally to avoid oversubscription.
The standalone Vulkan backend now supports row-major matrix multiplication and
fused matching-shape elementwise operations. Tensor forward matmul and common
elementwise activations can select this backend explicitly, while reductions,
softmax, attention, and gradient accumulation retain CPU fallbacks until their
dedicated kernels are added.

Predictor checkpoints include a tokenizer/model fingerprint and compact packed
region state. A mismatched vocabulary or model configuration is rejected before
parameters are installed, and successful saves are atomically renamed into
place. Grammar exceptions can be added through `GrammarEngine`; persistence of
learned grammar statistics will be added alongside the next checkpoint format.
