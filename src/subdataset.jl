"""
    SubDataset{D, T, <:AbstractDataset{D,T}, <:SubArray{<:SVector{D,T},1}} <: AbstractDataset{D,T}

A view of an `AbstractDataset`, as returned by the `view` function
or the `@view` macro on a dataset instance. A `SubDataset` is an `AbstractDataset`
of the same type as its parent, so indexing, iteration, and most other functions
can be expected to work in the same way for both the parent and the view.
"""
struct SubDataset{D, T, P<:AbstractDataset{D,T}, S<:SubArray{<:SVector{D,T},1}} <: AbstractDataset{D,T}
    parent::P
    data::S
    function SubDataset(par, data)
        @assert parent(data) === par.data
        P = typeof(par)
        S = typeof(data)
        SV = eltype(P)
        T = eltype(SV)
        D = length(SV)
        new{D,T,P,S}(par, data)
    end
end

function Base.summary(sd::SubDataset{D, T}) where {D, T}
    N = length(sd)
    return "$N-element view of $D-dimensional Dataset{$(T)}"
end

Base.parent(sd::SubDataset) = sd.parent
Base.parentindices(sd::SubDataset) = parentindices(sd.data)

"""
    view(d::Dataset, indices)

Return a view into the parent dataset `d`, as a [`SubDataset`](@ref)
that contains the datapoints of `d` referred to by `indices`.
"""
Base.view(d::AbstractDataset, i) = SubDataset(d, view(d.data, i))

function Base.view(::AbstractDataset, ::Any, ::Any, ::Vararg)
    throw(ArgumentError("Dataset views only accept indices on one dimension"))
end
