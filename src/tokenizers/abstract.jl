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
    merges::Dict{Tuple{I,I},I}

    special_tokens::Vector{I}

    function Tokenizer(args...; kwargs...)
        return Tokenizer{UInt16}(args...; kwargs...)
    end

    function Tokenizer{I}(; alphabet_size=255) where {I<:Integer}
        vocabulary_data = [UInt8(i) for i in 0:alphabet_size]
        vocabulary_offsets = [Int(i) for i in 1:(alphabet_size + 1)]
        vocabulary_lengths = ones(Int, alphabet_size)

        merges = Dict{Tuple{I,I},I}()

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

"""
    data(t::Tokenizer) = t.vocabulary_data
"""
data(t::Tokenizer) = t.vocabulary_data

"""
    offset(t::Tokenizer) = t.vocabulary_offsets
"""
offset(t::Tokenizer) = t.vocabulary_offsets

"""
    lengths(t::Tokenizer) = t.vocabulary_lengths
"""
lengths(t::Tokenizer) = t.vocabulary_lengths

"""
    merges(t::Tokenizer) = t.merges
"""
merges(t::Tokenizer) = t.merges

"""
    special_tokens(t::Tokenizer) = t.special_tokens
"""
special_tokens(t::Tokenizer) = t.special_tokens


"""
    function token(t::Tokenizer, i::Int)

Return the bytes associated with the i-th token ID.
"""
function token(t::Tokenizer, i::Int)
    _offset_i = offset(t)[i]
    return view(data(t), _offset_i:(_offset_i + lengths(t)[i] - 1))
end

"""

"""
struct BasicTokenizer{I<:Integer}
    tokenizer::Tokenizer{I}

    function BasicTokenizer(args..., kwargs...)
        return BasicTokenizer{UInt16}(args...; kwargs...)
    end

    function BasicTokenizer{I}(; alphabet_size=255) where {I<:Integer}
        return new{I}(Tokenizer(; alphabet_size=alphabet_size))
    end
end

"""
"""
function train(
    bt::BasicTokenizer{I}, vocabulary_size::Int, documents::Vector{String}
) where {I<:Integer}
    tokenizer = bt.tokenizer

    @assert vocabulary_size >= length(tokenizer.vocabulary_data)

    num_merges = vocabulary_size - length(tokenizer.vocabulary_data)

    # ----------------------------
    # concatenate + UTF-8 bytes
    # ----------------------------
    # equivalent to Python: text.encode("utf-8")
    text_bytes = Vector{UInt8}()
    for doc in documents
        append!(text_bytes, codeunits(doc))
    end

    # ids = list(text_bytes)
    ids = Vector{I}(text_bytes)

    merges = Dict{Tuple{I,I},I}()
    vocab = Dict{I,Vector{UInt8}}()

    # initialize vocab from existing tokenizer state
    for i in eachindex(tokenizer.vocabulary_offsets)
        start = tokenizer.vocabulary_offsets[i]
        len = tokenizer.vocabulary_lengths[i]
        vocab[I(i - 1)] = tokenizer.vocabulary_data[start:(start + len - 1)]
    end

    # ----------------------------
    # BPE merge loop
    # ----------------------------
    for i in 1:num_merges
        stats = get_stats(ids)  # Dict{Tuple{I,I}, Int}

        pair = findmax(stats)[2]  # most frequent pair

        new_id = I(length(tokenizer.vocabulary_data) + i)

        ids = merge_ids(ids, pair, new_id)

        merges[pair] = new_id
        vocab[new_id] = vcat(vocab[pair[1]], vocab[pair[2]])

        if verbose
            println(
                "merge $i/$num_merges: $pair -> $new_id (",
                vocab[new_id],
                ") had ",
                stats[pair],
                " occurrences",
            )
        end
    end

    # ----------------------------
    # store results in tokenizer
    # ----------------------------
    tokenizer.merges = merges

    # rebuild flattened vocab representation
    tokenizer.vocabulary_data = UInt8[]
    tokenizer.vocabulary_offsets = Int[]
    tokenizer.vocabulary_lengths = Int[]

    for (id, bytes_seq) in sort(collect(vocab); by=x -> x[1])
        push!(
            tokenizer.vocabulary_offsets, length(tokenizer.vocabulary_data) + 1
        )
        push!(tokenizer.vocabulary_lengths, length(bytes_seq))
        append!(tokenizer.vocabulary_data, bytes_seq)
    end

    return bt
end
