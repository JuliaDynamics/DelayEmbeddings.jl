using LinearAlgebra
using Plots

test = sin(1:1000).*exp(-0.005*1:1000)

plot(test)
