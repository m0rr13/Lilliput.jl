# Lilliput.jl
My trial at implementing a transformer architecture, starting from "Build a Large Language Model (From Scratch)" by Sebastian Raschka, and using as few dependencies as possible.

### Tokenizer

I built three tokenizers extending the common `AbstractTokenizer` interface: `BasicTokenizer`, `RegexTokenizer` and `GPT4Tokenizer`.

Resources:
- [Karpathy minbpe](https://github.com/karpathy/minbpe/).
