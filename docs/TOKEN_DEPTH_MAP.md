# Token depth map

The initial map is [token-depth-regions.tsv](../data/token-depth-regions.tsv).
It is a policy file loaded by `LayeredTokenizer` for the initial byte layer.
Its mobility values now scale adaptive tokenizer-node updates; a deeper token
therefore changes its learned prototypes more slowly.

For now, each future token receives two values from its region:

- `depth`: 0 means shallow and easy to change; 255 means deep and resistant to
  change.
- `mobility`: an adaptive-tokenizer training multiplier. A shallow token at
  `1.00` receives its full update, while a deep alphabetic token at `0.12`
  receives 12 percent of that update.

The byte ranges are only a temporary bootstrap layout. A word-piece, BPE, or
other tokenizer can reuse the same semantic regions while assigning its own
token IDs.
