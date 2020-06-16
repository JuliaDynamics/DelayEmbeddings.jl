using DynamicalSystemsBase

lo = Systems.lorenz()

tr = trajectory(lo, 100; dt = 0.1, Ttr = 10)

x = tr[:, 1]

cost = uzal_cost(x, whatever)
