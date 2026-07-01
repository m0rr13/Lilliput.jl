module Tokenizers

include("utilities.jl")
export count_consecutives

include("abstract.jl")
export AbstractTokenizer
export train, encode, decode
export Tokenizer

end
