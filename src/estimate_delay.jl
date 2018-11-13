using StatsBase: autocor
# using LsqFit: curve_fit
export estimate_delay

#####################################################################################
#                               Estimate Delay Times                                #
#####################################################################################
"""
    estimate_delay(s, method::String) -> τ

Estimate an optimal delay to be used in [`reconstruct`](@ref) or [`embed`](@ref).
Return the exponential decay time `τ` rounded to an integer.

The `method` can be one of the following:

* `"first_zero"` : find first delay at which the auto-correlation function becomes 0.
* `"first_min"` : return delay of first minimum of the auto-correlation function.
"""
function estimate_delay(x::AbstractVector, method::String; maxtau=100, k=1)
    method ∈ ("first_zero", "first_min") ||
        throw(ArgumentError("Unknown method"))

    if method=="first_zero"
        c = autocor(x, 0:length(x)÷10; demean=true)
        i = 1
        # Find 0 crossing:
        while c[i] > 0
            i += 1
            i == length(c) && break
        end
        return i

    elseif method=="first_min"
        c = autocor(x, 0:length(x)÷10, demean=true)
        i = 1
        # Find min crossing:
        while c[i+1] < c[i]
            i+= 1
            i == length(c)-1 && break
        end
        return i
    # elseif method=="exp_decay"
    #     c = autocor(x, demean=true)
    #     # Find exponential fit:
    #     τ = exponential_decay(c)
    #     return round(Int,τ)
    # elseif method=="mutual_inf"
    #     m = mutinfo(k, x, x)
    #     L = length(x)
    #     for i=1:maxtau
    #         n = mutinfo(k, view(x, 1:L-i), view(x, 1+i:L))
    #         n > m && return i
    #         m = n
    #     end
    end
end


# Here is the code that does exponential decay:
# """
#     localextrema(y) -> max_ind, min_ind
# Find the local extrema of given array `y`, by scanning point-by-point. Return the
# indices of the maxima (`max_ind`) and the indices of the minima (`min_ind`).
# """
# function localextrema end
# @inbounds function localextrema(y)
#     l = length(y)
#     i = 1
#     maxargs = Int[]
#     minargs = Int[]
#     if y[1] > y[2]
#         push!(maxargs, 1)
#     elseif y[1] < y[2]
#         push!(minargs, 1)
#     end
#
#     for i in 2:l-1
#         left = i-1
#         right = i+1
#         if  y[left] < y[i] > y[right]
#             push!(maxargs, i)
#         elseif y[left] > y[i] < y[right]
#             push!(minargs, i)
#         end
#     end
#
#     if y[l] > y[l-1]
#         push!(maxargs, l)
#     elseif y[l] < y[l-1]
#         push!(minargs, l)
#     end
#     return maxargs, minargs
# end

# function exponential_decay_extrema(c::AbstractVector)
#     ac = abs.(c)
#     ma, mi = localextrema(ac)
#     # ma start from 1 but correlation is expected to start from x=0
#     ydat = ac[ma]; xdat = ma .- 1
#     # Do curve fit from LsqFit
#     model(x, p) = @. exp(-x/p[1])
#     decay = curve_fit(model, xdat, ydat, [1.0]).param[1]
#     return decay
# end
#
# function exponential_decay(c::AbstractVector)
#     # Do curve fit from LsqFit
#     model(x, p) = @. exp(-x/p[1])
#     decay = curve_fit(model, 0:length(c)-1, abs.(c), [1.0]).param[1]
#     return decay
# end
