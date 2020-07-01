## test some julia stuff
using StatsBase
using Distances

n = 4

a = collect(1:9)

data_sample = sample(1:9, n; replace=false)


fld(7.9999,1)

T = 40
Y = tr
NN = length(Y)-T;

SampleSize = .5
NNN = fld(SampleSize*NN,1)

data_sample = sample(1:NN, NNN; replace=false)

floor(Int,4.5)

A = [120 340 44]

for i in A
    display(i)
end

metric = Chebyshev()
metric = Euclidean()
# play around with pairwise distances
A = [1 2 1]
B = [3 4 3]
C = [2 2 2]
TT = vcat(A,B,C)

pdd = pairwise(metric,TT, dims = 1)

for i = 1:20
    display(i)
end

evaluate.(metric,TT,B)
K = 3
neighborhood = zeros(K+1,size(tr,2))

idx = [1 2 3]

neighborhood[2:K+1,:] = tr[idx]...

tr[idx]

AA = tr[idx]
BB = hcat(AA...)

neighborhood[2:K+1,:] = hcat(tr[idx]...)
