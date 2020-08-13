# v1.9.0
* New `uzal_cost_local`: Local version of `uzal_cost` function.
# v1.8.0
* New cost function for testing "goodness" of a delay embedding: `uzal_cost` from Uzal et. al.
# v1.7.0
* New function `regularize`, useful in doing dataset-related operations (e.g. finding fractal dimension)
# v1.6.1
* `genembed` and co. now allow input arguments as vectors.
# v1.6
* Implemented efficient `eachcol` and `eachrow` iterators on Datasets

# v1.5
* Unified approach to delay embeddings by Pecora et al implemented.
# v1.4.3
* `js` of `genembed` defaults to ones.
* low level version of `getembed` takes in an embedding.

# v1.4
* New generalized embeddings: `genembed`, `GeneralizedEmbedding` that can do any combination possible from a dataset.
* Using `SizedArrays` in multidimensional embeddings is deprecated and will be removed in later versions.
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
