using Documenter
using Lilliput

makedocs(;
    sitename="Lilliput.jl documentation", modules=[Lilliput], remotes=nothing, doctest=true
)
