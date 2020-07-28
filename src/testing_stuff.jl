using Distances

ϵ_ball = zeros(3, 3) # preallocation
ϵ_ball[1,:] = [2; 2; 1]
ϵ_ball[2,:] = [4; 2; 5]
ϵ_ball[3,:] = [1; 3; 7]

u_k = sum(ϵ_ball,dims=1) ./ 3

E²_sum = sum((evaluate(metric,ϵ_ball,u_k)).^2)

A = [2 2 1];
B = [4 2 5];
C = [1 3 7];


evaluate(metric,ϵ_ball,repeat(u_k,3,1))

colwise(metric,ϵ_ball,repeat(u_k,3,1))

pairwise(metric,ϵ_ball,u_k)


repeat(u_k,3,1)

for (i,τ) in enumerate(0:10)
    display(i)
    display(τ)
end

test = [1; 2; 3; 4; 5]

tt = embed(test,size(test,2)+1,1)

ttt = embed2(tt,test,1)

Y = tt

τ = 1
N = size(Y,1)   # length of input trajectory
NN = size(Y,2)  # dimensionality of input trajectory
M = N - τ
Y_new = zeros(M,NN+1)

Y_new[:,1:NN] .= Y[1:M,:]

Y_new[:,1:2] .= collect(Y[1:M,1:2])

hcat(Y[1:M,1:2]...)


ABC = fill(Int[], 1, 3)

ABC[1] = [1,2,3]

ABC[2] = [3;4]

D = pairwise(Euclidean(),Matrix(tt),Matrix(tt), dims = 1)

D + Matrix(I,size(D))*9999

DD = Matrix(I,4,4)

test_A = [1 23 7 0; 8 7 16 1]
idxs = mapslices(sortperm, test_A, dims=2)

idx2 = LinearIndices(idxs)
idx3 = CartesianIndices(idxs[1,:])

ind = LinearIndices(size(test_A),1:2,idxs)

A = test_A


display(hcat([sortperm(A[i,:]) for i=1:size(A,1)]...))


@time begin
idxs = mapslices(sortperm, test_A, dims=2)
end
@time begin
idxs2= hcat([sortperm(A[i,:]) for i=1:size(A,1)]...)
end

CartesianIndex.(transpose(hcat([i*ones(Int,size(A,2)) for i=1:size(A,1)]...)),
        mapslices(sortperm, test_A, dims=2))

Y_temp = rand(50,3)
Y_temp = Dataset(Y_temp)
NN = length(Y_temp)
T = 5
N = NN - T
w = 2
metric = Euclidean()
vtree = KDTree(Y_temp[1:N], metric) # tree for input data
allNNidxs, d_E1 = all_neighbors(vtree, Y_temp[1:N], (1:N), 1, w)

newfididxs = [i .+ T for i = 1:N]
newNNidxs = [i .+ T for i in allNNidxs]

d_E2 = [evaluate(metric,Y_temp[newfididxs[i]],Y_temp[newNNidxs[i][1]]) for i = 1:N]


test = [4; 3; 7]


sum(test.<3)

NN_distances = fill(Int[], 1, τ_max+1)
NN_distances = fill(AbstractArray[], 1, τ_max+1)
NN_distances[1] = ts

NN_distances = [AbstractArray[] for i=1, j=1:τ_max+1]

push!(NN_distances[1],ts)

NN_distances[2]

###
lo = Systems.lorenz()

tr = trajectory(lo, 60; dt = 0.01, Ttr = 10)

Y = tr

s = Y[:,1]

metric = Euclidean()
K = 1
w = 1
τ_max = 50
N = length(Y)
NN = N-τ_max
vtree = KDTree(Y[1:NN], metric)
allNNidxs, Δx = DelayEmbeddings.all_neighbors(vtree, Y[1:NN], 1:NN, K, w)

# loop over all phase space points in order to compute Δϕ
Δϕ = zeros(NN,τ_max+1)     # preallocation
for j = 1:NN
    # loop over all considered τ's
    for (i,τ) in enumerate(0:τ_max)
        Δϕ[j,i] = abs(s[j+τ]-s[allNNidxs[j][1]+τ]) / Δx[j][1] # Eq. 14 & 15
    end
end

##
##

s = Y[:,1]

pop!(s)
ss = Dataset(s)
s2 = Dataset(s1)

function test_fun(Y::Dataset,s::Dataset)
    @assert length(s)>=length(Y) "The length of the input time series `s` must be at least the length of the input trajectory `Y` "
end
function test_fun(Y::Dataset,s::Array)
    @assert length(s)>=length(Y) "The length of the input time series `s` must be at least the length of the input trajectory `Y` "
end

test_fun(Y,ss)

test_matrix = [1 2 3 4; 2 11 8 9]


####

#s = s[100:2500]
Y = Dataset(s)

# theiler window
theiler = 57
tau_max = 100

metric = Euclidean()    # consider only Euclidean norm
K = 1                   # consider only first nearest neighbor
N = length(Y)           # length of the phase space trajectory
NN = N - τ_max          # allowed length of the trajectory w.r.t. τ_max

# tree for input data
vtree = KDTree(Y[1:NN], metric)
# compute nearest neighbors
allNNidxs, Δx = DelayEmbeddings.all_neighbors(vtree, Y[1:NN], 1:NN, K, w)   # Eq. 12

# loop over all phase space points in order to compute Δϕ
Δϕ = zeros(NN,τ_max+1)     # preallocation
for j = 1:NN
    # loop over all considered τ's
    for (i,τ) in enumerate(0:τ_max)
        Δϕ[j,i] = abs(s[j+τ][1]-s[allNNidxs[j][1]+τ][1]) / Δx[j][1] # Eq. 14 & 15
    end
end

# compute final beta statistic
β = mean(log10.(Δϕ), dims=1)

test = log10.(Δϕ)


for i = 1:length(Δx)
    if Δx[i][1] == 0
        display(i)
    end
end
