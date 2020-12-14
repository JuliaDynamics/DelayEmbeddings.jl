# v1.19.0
* Theiler window is now usable in Cao's method.

# v1.18.0
* `view` is now applicable to `AbstractDataset`, producing objects of the new type `SubDataset`.

# v1.17.0
* All code related to neighborhoods and finding nearest neighbors has moved to Neighborhood.jl, and thus old names like `FixedMassNeighborhood` and `neighborhood` have been deprecated.
* `mutualinformation` is deprecated in favor of `selfmutualinfo`.

# v1.16.0
* Arbitrary weights can be given as options to `genembed`.

# v1.15.0
* Horizontal concatenation of same-length `Vector{<:Real}` and `Dataset` in any order using
  `Base.hcat(x, y)` or `[x y]` syntax.
* Convenience constructors that uses horizontal concatenation:
  `Dataset(::Dataset, ::Vector{<:Real})`, `Dataset(::Vector{<:Real}, ::Dataset)` and
  `Dataset(::Dataset, ::Dataset)`.

# v1.14.0
* New unified embedding method `pecuzal` by Kraemer et al.

# v1.13.0
* `reconstruct` is deprecated in favor of `embed`.

# v1.12.0
* Possible delay times in `optimal_traditional_de` are now `1:100` for increased accuracy.
* New method for univariate non-unified delay embedding by Hegger, Kantz
* It is now possible to `embed` in one dimension (which just returns the vector as a Dataset)
* New function `optimal_traditional_de` for automated delay embeddings
* `delay_afnn, delay_ifnn, delay_fnn, delay_f1nn` are part of public API now.
* The argument `γs` and the function `reconstruct` is starting to be phased out in
  favor of `ds` and `embed`.
* Multi-timeseries via `reconstruct` or `embed` is deprecated in favor of using `genembed`.

## Deprecations
* Using `estimate_dimension` is deprecated in favor of either calling `afnn, fnn, ...` directly or using the function `optimal_traditional_de`

# v1.11.0
* Dropped RecipesBase
* Now exports `dimension` (there was a problem where we have forgotten to export `dimension` from here, and `DynamicalSystemsBase` was overwritting it)

# v1.10.0
* New methods (Garcia & Almeida and Nichkawde/MDOP) for unified delay coordinates
  now available for multivariate input data.
* New method (Garcia & Almeida) for unified delay coordinates embeddings.
* Inner function `findlocalminima`, that can be used on cycle-by-cycle optimal embedding creation.

# v1.9.0
* New `uzal_cost_local`: Local version of `uzal_cost` function.
* Chetan Nichkawde method for unified delay coordinates embedding.
* New `hcat` function for delay coordinates construction: `hcat_lagged_values`

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
