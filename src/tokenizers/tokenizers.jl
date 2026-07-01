module Tokenizers

include("utilities.jl")
export count_consecutives, merge

include("abstract.jl")
export AbstractTokenizer
export train, encode, decode
export Tokenizer
export data, offset, lengths, merges, special_tokens



end
