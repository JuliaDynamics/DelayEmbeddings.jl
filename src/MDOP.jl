using Neighborhood
using StatsBase
using Distances

export beta_statistic
export MDOP


"""
    MDOP(....)
method by [^Nichkawde2013]


[^Nichkawde2013]: Nichkawde, Chetan (2013). [Optimal state-space reconstruction using derivatives on projected manifold. Physical Review E 87, 022905](https://doi.org/10.1103/PhysRevE.87.022905).
"""
function MDOP(Y...)
    #tbd

end


"""
    beta_statistic(Y::Dataset, s::Dataset(::Array); kwargs...) → β
Compute the β-statistic `β` for input phase space trajectory `Y` (of type
`Dataset`) and a univariate time series 's' (1-dim. `Array` or `Dataset`)
according to Nichkawde [^Nichkawde2013], based on estimating derivatives on a
projected manifold. For a range of delay values `τ` (ranging from `0:1:τ_max`)
`β` gets computed and its maximum over all considered `τ`'s serves as the
optimal delay considered in this embedding cycle.

## Keyword arguments

* `τ_max = 50`: Maximum value of considered delay values `τ` (in sampling time
  units), ranging from `τ=0:1:τ_max`. For each of the `τ`'s the β-statistic gets
  computed.
* `w = 1`: Theiler window (neighbors in time with index `w` close to the point,
  that are excluded from being true neighbors). `w=0` means to exclude only the
  point itself, and no temporal neighbors.

## Description
The `β`-statistic is based on the geometrical idea of maximal unfolding of the
reconstructed attractor and is tightly related to the False Nearest Neighbor
method ([^Kennel1992]). In fact the method eliminates the maximum amount of
false nearest neighbors in each embedding cycle. The idea is to estimate the
absolute value of the directional derivative with respect to a possible new
dimension in the reconstruction process, and with respect to the nearest
neighbor, for all points of the phase space trajectory:

ϕ'(τ) = Δϕ_d(τ) / Δx_d

Δx_d is simply the Euclidean nearest neighbor distance for a reference point
with respect to the given Theiler window `w`. Δϕ_d(τ) is the distance of the
reference point to its nearest neighbor in the one dimensional time series `s`,
for the specific τ. Δϕ_d(τ) = |s(i+τ)-s(j+τ)|, with i being the index of the
considered reference point and j the index of its nearest neighbor.

Finally,

`β` = log β(τ) = ⟨log ϕ'(τ)⟩ ,

with ⟨.⟩ being the mean over all reference points. When one chooses the maximum
of `β` over all considered τ's, one obtains the optimal delay value for this
embedding cycle. Note that in the first embedding cycle, the input phase space
trajectory `Y` can also be just a univariate time series.

[^Nichkawde2013]: Nichkawde, Chetan (2013). [Optimal state-space reconstruction using derivatives on projected manifold. Physical Review E 87, 022905](https://doi.org/10.1103/PhysRevE.87.022905).
[^Kennel1992]: Kennel, M. B., Brown, R., Abarbanel, H. D. I. (1992). [Determining embedding dimension for phase-space reconstruction using a geometrical construction. Phys. Rev. A 45, 3403] (https://doi.org/10.1103/PhysRevA.45.3403).
"""

function beta_statistic()
    # tbd

end
