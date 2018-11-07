using Yao, Yao.Boost, Yao.Intrinsics, StaticArrays, Yao.Blocks
using CuYao
using BenchmarkTools

nbit = 5
c = chain(put(nbit, 2=>X), put(nbit, 2=>rot(X, 0.2)), control(nbit, 3, 2=>rot(X,0.3)))
cc = c |> KernelCompiled
reg = rand_state(nbit) |> cu

@benchmark $reg |> copy |> $(c[1]) seconds = 2
@benchmark reg |> copy |> cc seconds = 2
