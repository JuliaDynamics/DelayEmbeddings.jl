# v1.3
* Allow indexing datasets with boolean vectors.
# v1.2.0
- New embedding method `WeightedDelayEmbedding`, which does the same as `DelayEmbedding` but further weights the entries of the embedded space by a weight `w^γ` for each `γ`. See the updated docstring of `reconstruct`.
# v1.1.0
- Added a recipe to plot `Datasets` as `Matrices` in `Plots.jl`

# v1.0.0
Nothing changed from the previous version. This is just the official 1.0 release in accordance with SemVer. :)

# v0.3.0

- Added a new and superior method to compute mutual information, based on the algorithm of Fraser
- Added method that uses mutual information in `estimate_delay`
- Added method for exponential decay in `estimate_delay`
- Added metric choice support for `estimate_dimension`
