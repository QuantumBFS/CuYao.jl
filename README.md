# CuYao.jl

[![codecov](https://codecov.io/gh/QuantumBFS/CuYao.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/QuantumBFS/CuYao.jl)

GPU support to [Yao.jl](https://github.com/QuantumBFS/Yao.jl).

We are in an early-release beta. Expect some adventures and rough edges.

## Installation

In **v1.0+**, please type `]` in the REPL to use the package mode, then type:

```julia
pkg> add Yao
pkg> add CuYao
```


## Documentation
It provides only two new APIs, `reg |> cu` to upload a quantum register to GPU, and `cureg |> cpu` to download a quantum register to CPU.

To start, see the following example
```julia
using Yao, CuYao

cureg = rand_state(9, 1000) |> cu 
cureg |> put(9, 2=>Z)
measure!(cureg |> addbit(1) |> focus!(4,1,3)) |> relax!
```

## Features
A List of Supported Gates
- [x] general U(N) gate
- [x] general U(1) gate
- [x] better X, Y, Z gate
- [x] better T, S gate
- [x] SWAP gate
- [x] better control gates
- [x] BP diff blocks

Supported Register Operations
- [x] measure!, measure_reset!, measure_remove!, select
- [x] addbit!
- [x] insert_qubit!
- [x] focus!, relax!
- [x] join
- [x] density_matrix

Other Operations
- [x] statistic functional diff blocks
- [x] expect for statistic functional

If you find an unsupported useful feature, welcome to file an issue or submit a PR.

## Contribution

To contribute to this project, please open an [issue](https://github.com/QuantumBFS/CuYao.jl/issues) first to discuss with us in case we may not accept your PR.

## Author

This project is an effort of QuantumBFS, an open source organization for quantum science. All the contributors are listed in the [contributors](https://github.com/QuantumBFS/CuYao.jl/graphs/contributors).

## License

**CuYao** is released under the Apache 2 license.
