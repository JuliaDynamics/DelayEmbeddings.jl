# v2.6 - Refactoring release

All functionality related to `StateSpaceSet` has been refactored into a new package StateSpaceSets.jl. Since DelayEmbeddings.jl re-exports it, nothing should be breaking, but still, if you don't explicitly need delay embeddings, you should use StateSpaceSets.jl directly.

Now DelayEmbeddings.jl really is only about delay coordinate embedding methods.

The package has also been updated to all new names used in DynamicalSystems.jl v3.0.

Nothing else has been changed.

# v2.5
- `statespace_sampler` ported here from ChaosTools.jl

# v2.4
- It is now possible to horizontally concatenate more than two `StateSpaceSet`s using `hcat`. Providing multiple `StateSpaceSet`s of potentially different dimensions to the `StateSpaceSet` constructor will horizontally concatenate the inputs.

# v2.3
- New function `dataset_distance` that calculates distances between datasets.
- New method/metric `Hausdorff` that can be used in `dataset_distance`.
- New function `datasets_sets_distances` that calculates distances between sets of datasets.

# v2.2.0
* added option for different return type for orthonormal function, returns now SMatrix only for small matrices, otherwise Matrix

# v2.0.0
**BREAKING**
- All deprecations have been removed and errors will be thrown now instead. Switch to previous stable version to enable them again.
- `StateSpaceSet[range_of_integers]` now returns a `StateSpaceSet` instead of `Vector{SVector}`.

# v1.20.6
* Name `regularize` has been deprecated in favor of `standardize`, which aligns more with current literature.

# v1.20.1
* Patch for correct keyword arguments in PECUZAL algorithm

# v1.20.0
* Revised version of PECUZAL algorithm, now tracking maximal L-decrease

# v1.19.3
* Automated traditional delay embedding has improved clause for Cao's method.
# v1.19.0
* Theiler window is now usable in Cao's method.

# v1.18.0
* `view` is now applicable to `AbstractStateSpaceSet`, producing objects of the new type `SubDataset`.

# v1.17.0
* All code related to neighborhoods and finding nearest neighbors has moved to Neighborhood.jl, and thus old names like `FixedMassNeighborhood` and `neighborhood` have been deprecated.
* `mutualinformation` is deprecated in favor of `selfmutualinfo`.

# v1.16.0
* Arbitrary weights can be given as options to `genembed`.

# v1.15.0
* Horizontal concatenation of same-length `Vector{<:Real}` and `StateSpaceSet` in any order using
  `Base.hcat(x, y)` or `[x y]` syntax.
* Convenience constructors that uses horizontal concatenation:
  `StateSpaceSet(::StateSpaceSet, ::Vector{<:Real})`, `StateSpaceSet(::Vector{<:Real}, ::StateSpaceSet)` and
  `StateSpaceSet(::StateSpaceSet, ::StateSpaceSet)`.

# v1.14.0
* New unified embedding method `pecuzal` by Kraemer et al.

# v1.13.0
* `reconstruct` is deprecated in favor of `embed`.

# v1.12.0
* Possible delay times in `optimal_traditional_de` are now `1:100` for increased accuracy.
* New method for univariate non-unified delay embedding by Hegger, Kantz
* It is now possible to `embed` in one dimension (which just returns the vector as a StateSpaceSet)
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
