using Yao, CuYao, CUDA
using BenchmarkTools

reg = rand_state(12; nbatch=1000)
creg = reg |> cu
@benchmark CUDA.@sync creg |> put(12, 3=>Z)
@benchmark CUDA.@sync creg |> put(12, 3=>X)
@benchmark reg |> put(12, 3=>Z)
@benchmark CUDA.@sync creg |> control(12, 6, 3=>X)
@benchmark reg |> control(12, 6, 3=>X)
@benchmark CUDA.@sync creg |> put(12, 3=>rot(X, 0.3))
@benchmark reg |> put(12, 3=>rot(X, 0.3))

reg = rand_state(20)
creg = reg |> cu
g = swap(12, 7, 2)
@benchmark reg |> g
@benchmark CUDA.@sync creg |> g
