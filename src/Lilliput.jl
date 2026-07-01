module Lilliput

include("tokenizers/tokenizers.jl")
using .Tokenizers

# yep, I am not using Reexport
export count_consecutives, merge
export AbstractTokenizer
export train, encode, decode
export Tokenizer
export data, offsets, bytelengths, merges, special_tokens
export vocabulary_lastindex, token, add_token!
export BasicTokenizer

end
