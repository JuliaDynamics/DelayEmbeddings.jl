cd(@__DIR__)

using DelayEmbeddings

import Downloads
Downloads.download(
    "https://raw.githubusercontent.com/JuliaDynamics/doctheme/master/build_docs_with_style.jl",
    joinpath(@__DIR__, "build_docs_with_style.jl")
)
include("build_docs_with_style.jl")

pages = [
    "index.md",
    "embed.md",
    "separated.md",
    "unified.md",
]

build_docs_with_style(pages, DelayEmbeddings, StateSpaceSets;
    authors = "George Datseris <datseris.george@gmail.com>, Hauke Kraemer",
    expandfirst = ["index.md"], #  this is the first script that loads colorscheme
)
