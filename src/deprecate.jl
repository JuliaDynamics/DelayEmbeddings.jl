export mutualinformation
function mutualinformation(args...; kwargs...)
    @warn "`mutualinformation` is deprecated in favor of `selfmutualinfo`."
    selfmutualinfo(args...; kwargs...)
end
