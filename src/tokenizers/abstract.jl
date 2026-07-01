abstract type AbstractTokenizer end

"""
    function train(
        abstract_tokenizer::AbstractTokenizer,
        vocabulary_size::Int,
        documents::Vector{String},
    )

Train a vocabulary of size `vocabulary_size` from the given `text`.
"""
function train(
    abstract_tokenizer::AbstractTokenizer,
    vocabulary_size::Int,
    documents::Vector{String},
)
    throw(MethodError(train, (abstract_tokenizer, vocabulary_size, documents)))
end

"""
    function encode(at::AbstractTokenizer, text::String)
        throw(MethodError(encode, (at, text,)))
    end

Encode the given `text` as a list of text IDs (integers).
"""
function encode(at::AbstractTokenizer, text::String)::Vector{<:Integer}
    throw(MethodError(encode, (at, text)))
end

"""
    function decode(at::AbstractTokenizer, alphabet_size::Vector{<:Integer})
        throw(MethodError(decode, (at, alphabet_size)))
    end

Decode a list of integers into a string.
"""
function decode(at::AbstractTokenizer, alphabet_size::Vector{<:Integer})::String
    throw(MethodError(decode, (at, alphabet_size)))
end

"""
    struct Tokenizer{I<:Integer} <: AbstractTokenizer
        vocabulary_data::Vector{UInt8}
        vocabulary_offsets::Vector{Int}
        vocabulary_lengths::Vector{Int}

        merges::Dict{Tuple{I,I}, I}

        special_tokens::Vector{I}
    end
"""
struct Tokenizer{I<:Integer} <: AbstractTokenizer
    # flattened vocabulary, where each token is the following bytes sequence:
    # view(vocabulary_data, offsets[i] : offsets[i] + lengths[i] - 1)
    vocabulary_data::Vector{UInt8}
    vocabulary_offsets::Vector{Int}
    vocabulary_lengths::Vector{Int}

    # two token ids i,j, are merged in merges[(i,j,)]
    merges::Dict{Tuple{I,I}, I}

    special_tokens::Vector{I}

    function Tokenizer(args...; kwargs...)
        return Tokenizer{UInt16}(args...; kwargs...)
    end

    function Tokenizer{I}(; alphabet_size=255) where {I<:Integer}
        vocabulary_data = [UInt8(i) for i in 0:alphabet_size]
        vocabulary_offsets = [Int(i) for i in 1:(alphabet_size + 1)]
        vocabulary_lengths = ones(Int, alphabet_size)

        merges = Dict{Tuple{I,I}, I}()

        special_tokens = I[]

        return new{I}(
            vocabulary_data,
            vocabulary_offsets,
            vocabulary_lengths,
            merges,
            special_tokens,
        )
    end
end
