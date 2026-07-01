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
        vocabulary_bytelengths::Vector{Int}

        merges::Dict{Tuple{I,I}, I}

        special_tokens::Vector{I}
    end

See also [`BasicTokenizer`](@ref).
"""
struct Tokenizer{I<:Integer} <: AbstractTokenizer
    # flattened vocabulary, where each token is the following bytes sequence:
    # view(vocabulary_data, offsets[i] : offsets[i] + bytelengths[i] - 1)
    vocabulary_data::Vector{UInt8}
    vocabulary_offsets::Vector{Int}
    vocabulary_bytelengths::Vector{Int}

    # two token ids i,j, are merged in merges[(i,j,)]
    merges_data::Dict{Tuple{I,I},I}

    special_tokens::Vector{I}

    function Tokenizer(args...; kwargs...)
        return Tokenizer{UInt16}(args...; kwargs...)
    end

    function Tokenizer{I}(; alphabet_size=255) where {I<:Integer}
        vocabulary_data = [UInt8(i) for i in 1:alphabet_size]
        vocabulary_offsets = [Int(i) for i in 1:alphabet_size]
        vocabulary_bytelengths = ones(Int, alphabet_size)

        merges = Dict{Tuple{I,I},I}()

        special_tokens = I[]

        return new{I}(
            vocabulary_data,
            vocabulary_offsets,
            vocabulary_bytelengths,
            merges,
            special_tokens,
        )
    end
end

"""
    data(t::Tokenizer) = t.vocabulary_data
"""
data(t::Tokenizer) = t.vocabulary_data

"""
    offsets(t::Tokenizer) = t.vocabulary_offsets

Return a vector whose i-th index indicates where the i-th token is encoded,
within `data(t)`.

See also [`data(t::Tokenizer)`](@ref).
"""
offsets(t::Tokenizer) = t.vocabulary_offsets

"""
    bytelengths(t::Tokenizer) = t.vocabulary_bytelengths

Return a vector whose i-th index indicates how many bytes is long the i-th 
token encoded.
"""
bytelengths(t::Tokenizer) = t.vocabulary_bytelengths

"""
    merges(t::Tokenizer) = t.merges

Return the dictionary containing all the pair countings for BPE.
"""
merges(t::Tokenizer) = t.merges_data

"""
    special_tokens(t::Tokenizer) = t.special_tokens
"""
special_tokens(t::Tokenizer) = t.special_tokens

"""
    function vocabulary_lastindex(t::Tokenizer)

Return how many bytes are encoded within `data(t)`.

See also [`data(t::Tokenizer)`](@ref).
"""
function vocabulary_lastindex(t::Tokenizer)
    # or length(t.vocabulary_bytelengths)
    return length(t.vocabulary_offsets)
end

"""
    function token(t::Tokenizer, i::Int)

Return the bytes associated with the i-th token ID.
"""
function token(t::Tokenizer{I}, i::I) where {I}
    _offset_i = offsets(t)[i]
    return view(data(t), _offset_i:(_offset_i + bytelengths(t)[i] - 1))
end

"""
    function add_token!(t::Tokenizer, bytes::Vector{UInt8})

Add a new token to the tokenizer, by appending the bytes to the raw `data`
collection, and incrementing by one the size of the `byteslengths` and `offsets`
collections.

See also [`data(t::Tokenizer)`](@ref), [`bytelengths(t::Tokenizer)`](@ref),
[`offsets(t::Tokenizer)`](@ref).
"""
function add_token!(t::Tokenizer, bytes::Vector{UInt8})
    vocabulary_data = data(t)
    vocabulary_bytelengths = bytelengths(t)
    vocabulary_offsets = offsets(t)

    push!(vocabulary_bytelengths, length(bytes))
    push!(vocabulary_offsets, length(vocabulary_data)+1)
    append!(vocabulary_data, bytes)
end

"""
    struct BasicTokenizer{I<:Integer} <: AbstractTokenizer
        tokenizer::Tokenizer{I}
    end

Very simple implementation of an [`AbstractTokenizer`](@ref).

TODO: remove this and keep the `Tokenizer` only.
"""
struct BasicTokenizer{I<:Integer} <: AbstractTokenizer
    tokenizer::Tokenizer{I}

    function BasicTokenizer(args...; kwargs...)
        return BasicTokenizer{UInt16}(args...; kwargs...)
    end

    function BasicTokenizer{I}(; alphabet_size=255) where {I<:Integer}
        return new{I}(Tokenizer(; alphabet_size=alphabet_size))
    end
end

"""
    tokenizer(bt::BasicTokenizer) = bt.tokenizer
"""
tokenizer(bt::BasicTokenizer) = bt.tokenizer

"""
    function train(
        bt::BasicTokenizer{I}, vocabulary_size::Int, documents::Vector{String}
    ) where {I<:Integer}

Learn/train/build a vocabulary from the given documents, leveraging a BPE
implementation similar to the one of GPT-2.

See also:
- [the gpt-2 repository](https://github.com/openai/gpt-2/blob/master/src/encoder.py);
- [the Karphaty repository](https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py).

# Examples
```jldoctest
julia> using Lilliput

julia> bt = BasicTokenizer(); 

julia> train(bt, 270, ["Hello, world!", "Hello hello!"])

julia> Char.(token(bt.tokenizer, UInt16(256)))
2-element Vector{Char}:
 'l': ASCII/Unicode U+006C (category Ll: Letter, lowercase)
 'l': ASCII/Unicode U+006C (category Ll: Letter, lowercase)
```
"""
function train(
    bt::BasicTokenizer{I}, vocabulary_size::Int, documents::Vector{String}
) where {I<:Integer}
    _tokenizer = tokenizer(bt)

    vocabulary_data = data(_tokenizer)
    vocabulary_data_length = length(vocabulary_data)
    @assert vocabulary_size >= vocabulary_data_length "No space available"

    merges_data = merges(_tokenizer)
    num_merges = vocabulary_size - vocabulary_data_length

    # these are token IDs; now, they seem just bytes, byt are combined later 
    # in heavier integers. You want UInt16 here, probably UInt32.
    indexes = I[]
    for doc in documents
        append!(indexes, I.(codeunits(doc)))
    end

    for i in 1:num_merges
        counts = count_consecutives(indexes)

        # trying to call "findmax" later triggers an ArgumentError; just leave
        if isempty(counts)
            break
        end
        most_freqpair = findmax(counts)[2] # 2 returns a Tuple{I, I}

        # token ID for the new token
        new_id = I(vocabulary_lastindex(_tokenizer) + 1)

        # replace all the occurrences of the most frequent pair with the new ID
        indexes = merge(indexes, most_freqpair, new_id)

        merges_data[most_freqpair] = new_id

        add_token!(
            _tokenizer,
            vcat(
                token(_tokenizer, most_freqpair[1]),
                token(_tokenizer, most_freqpair[2]),
            ),
        )
    end

    return bt
end
